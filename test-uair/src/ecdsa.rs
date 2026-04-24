//! ECDSA signature-verification UAIR (minimal scalar slice).
//!
//! Spec source: `arithmetization_standalone/hybrid_arithmetics/ecdsa/ecdsa_intro.tex`.
//!
//! ## What this slice covers
//!
//! Only the scalar-field (`F_n`) and integer parts of the ECDSA arithmetization:
//!
//! 1. Bit-range: `b_i · (b_i − 1) == 0` for `i ∈ {1, 2}` at rows 0..255.
//! 2. Scalar bit-accumulation (mod n):
//!    `2·U_i[t] + b_i[t] − U_i[t+1] − q_{U_i}[t] · n == 0` at rows 0..255.
//! 3. Init boundary: `U_1[0] == 0`, `U_2[0] == 0`.
//! 4. Scalar inverse (mod n): `s · w − 1 − q_sw · n == 0` at row 256.
//! 5. Signature modular check: `x̂ − r − k · n == 0` with `k · (k − 1) == 0`.
//!
//! ## What it does NOT cover (deferred)
//!
//! - All F_p / elliptic-curve operation constraints (Jacobian doubling,
//!   addition, Shamir's trick, affine conversion). Those need a wider,
//!   degree-6 slice that hasn't been built yet.
//! - `u_1 = e · w`, `u_2 = r · w` boundary checks.
//! - Composition with the SHA-256 UAIR (binding `y_e` to a digest).
//!
//! ## F_n → Q[X] quotient-witness lift
//!
//! The spec places the scalar-accumulation and inverse constraints in
//! `F_n[X]`. The zinc-plus protocol has a single `Int` / `Fmod`
//! parameterization, so we re-express each `F_n` identity as an integer
//! identity with an explicit **quotient witness**:
//!
//! ```text
//!   2·U + b ≡ U_next   (mod n)   [spec, F_n]
//!   2·U + b = U_next + q · n     [Z / Q[X]]
//! ```
//!
//! The prover supplies `q` (bounded in `{0, 1}` for the accumulation
//! recurrence; `< n` for the inverse constraint). The verifier runs the
//! standard Q[X] `assert_zero` check; it does **not** verify `q < n`,
//! which is a soundness gap (lookup PIOP stubbed — same stance as the
//! SHA-256 slice).
//!
//! ## Soundness caveats
//!
//! - `q` ranges aren't enforced. A malicious prover could use large `q`.
//! - `k ∈ {0, 1}` is enforced arithmetically via `k · (k − 1) == 0`.
//! - `b_1, b_2 ∈ {0, 1}` likewise.
//! - `w_inv · s ≡ 1 (mod n)` is enforced, but not `w_inv < n`.
//! - `x̂` is a free witness in this slice (not tied to any EC computation).

use core::marker::PhantomData;

use crypto_bigint::{NonZero, Odd, Uint as CbUint};
use crypto_primitives::{
    ConstSemiring,
    crypto_bigint_int::Int,
};
use num_traits::Zero;
use rand::RngCore;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::dense::DensePolynomial,
};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, ShiftSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace,
    ideal::ImpossibleIdeal,
};

use crate::GenerateRandomTrace;

// ---------------------------------------------------------------------------
// Secp256k1 scalar-field order (n) and Int width used by this slice.
// ---------------------------------------------------------------------------

/// Number of 64-bit limbs used by the int columns. `Int<5>` is a 320-bit
/// signed integer — enough headroom to hold the 256-bit `n` and small
/// products (e.g. `q · n` with `q < 2^8`) without overflow.
pub const ECDSA_INT_LIMBS: usize = 5;

/// `n` = secp256k1 scalar-field order, padded to 320 bits (5 × 64).
///
/// Value: `n = 0xFFFF..FFFE BAAEDCE6AF48A03B BFD25E8CD0364141` (256-bit),
/// stored with one leading zero limb.
const SECP256K1_N_HEX: &str = concat!(
    "0000000000000000", // limb 4 (most significant, zero padding)
    "FFFFFFFFFFFFFFFF", // limb 3
    "FFFFFFFFFFFFFFFE", // limb 2
    "BAAEDCE6AF48A03B", // limb 1
    "BFD25E8CD0364141", // limb 0
);

/// Secp256k1 scalar-field order as a `crypto_bigint::Uint<5>`.
pub const SECP256K1_N_UINT: CbUint<ECDSA_INT_LIMBS> = CbUint::from_be_hex(SECP256K1_N_HEX);

/// Secp256k1 scalar-field order as a `crypto_primitives::Int<5>`.
pub const SECP256K1_N: Int<ECDSA_INT_LIMBS> =
    Int::new(crypto_bigint::Int::from_be_hex(SECP256K1_N_HEX));

// ---------------------------------------------------------------------------
// Column indices.
// ---------------------------------------------------------------------------

/// Column indices within the `int` slot. This slice uses **no** binary_poly
/// or arbitrary_poly columns; every value is a signed integer scalar.
pub mod cols {
    // Public int columns (prefix):
    pub const S_INIT: usize = 0;
    pub const S_ACCUM: usize = 1;
    pub const S_FINAL: usize = 2;
    pub const PA_E: usize = 3;
    pub const PA_R: usize = 4;
    pub const PA_S: usize = 5;
    /// Number of public int columns.
    pub const NUM_INT_PUB: usize = 6;

    // Witness int columns:
    pub const W_B1: usize = 6; // bit of u_1 at row t
    pub const W_B2: usize = 7; // bit of u_2 at row t
    pub const W_U1: usize = 8; // running accumulator for u_1
    pub const W_U2: usize = 9; // running accumulator for u_2
    pub const W_W_INV: usize = 10; // s^{-1} mod n (row 256)
    pub const W_XHAT: usize = 11; // integer R_x (row 256, free witness in this slice)
    pub const W_K: usize = 12; // quotient bit for sig check
    pub const W_Q_U1: usize = 13; // quotient for U_1 accumulation mod n
    pub const W_Q_U2: usize = 14; // quotient for U_2 accumulation mod n
    pub const W_Q_SW: usize = 15; // quotient for s · w − 1 mod n

    /// Total number of int columns.
    pub const NUM_INT: usize = 16;

    /// Flat trace indices for ShiftSpec. We use 0 binary_poly, 0 arbitrary_poly,
    /// so flat indexing coincides with `int` indexing here.
    pub const FLAT_W_U1: usize = W_U1;
    pub const FLAT_W_U2: usize = W_U2;
}

/// The trace row at which the final-state constraints apply.
///
/// Set to 256 (the spec's "row 257" — 0-indexed here). Rows 0..=255 carry
/// the bit-accumulation recurrence; row 256 is the scalar-inverse and
/// signature-modular-check row.
pub const FINAL_ROW: usize = 256;

// ---------------------------------------------------------------------------
// Scalar helpers (polynomial constants) — same pattern as sha256.rs.
// ---------------------------------------------------------------------------

/// Build a constant-polynomial (degree 0) `c` as a `DensePolynomial<R, 32>`.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// The UAIR.
// ---------------------------------------------------------------------------

/// ECDSA scalar-slice UAIR. See module docs for scope.
#[derive(Clone, Debug)]
pub struct EcdsaScalarSliceUair<R>(PhantomData<R>);

/// Minor trait: a `ConstSemiring` that can expose the secp256k1 scalar-field
/// order as one of its elements. Implemented only for `Int<ECDSA_INT_LIMBS>`
/// (the width this slice runs at).
pub trait EcdsaScalarRing: ConstSemiring + 'static {
    /// The scalar-field order `n`.
    fn secp256k1_n() -> Self;
}

impl EcdsaScalarRing for Int<ECDSA_INT_LIMBS> {
    fn secp256k1_n() -> Self {
        SECP256K1_N
    }
}

impl<R> Uair for EcdsaScalarSliceUair<R>
where
    R: EcdsaScalarRing,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, 0, cols::NUM_INT_PUB);
        let shifts: Vec<ShiftSpec> = vec![
            // W_U1 and W_U2 shifted by 1 so the accumulation constraint can
            // reach the next row's value.
            ShiftSpec::new(cols::FLAT_W_U1, 1),
            ShiftSpec::new(cols::FLAT_W_U2, 1),
        ];
        UairSignature::new(total, public, shifts, vec![])
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
        let s_accum = &int[cols::S_ACCUM];
        let s_final = &int[cols::S_FINAL];
        let pa_r = &int[cols::PA_R];
        let pa_s = &int[cols::PA_S];
        let b_1 = &int[cols::W_B1];
        let b_2 = &int[cols::W_B2];
        let u_1 = &int[cols::W_U1];
        let u_2 = &int[cols::W_U2];
        let w_inv = &int[cols::W_W_INV];
        let x_hat = &int[cols::W_XHAT];
        let k = &int[cols::W_K];
        let q_u1 = &int[cols::W_Q_U1];
        let q_u2 = &int[cols::W_Q_U2];
        let q_sw = &int[cols::W_Q_SW];

        // `down.int` layout (sorted by source_col, insertion order within ties):
        //   down.int[0] = up.W_U1 ^↓1
        //   down.int[1] = up.W_U2 ^↓1
        let down_u_1_sh1 = &down.int[0];
        let down_u_2_sh1 = &down.int[1];

        // Constant scalar `n` (the secp256k1 scalar-field order).
        let n_scalar = const_scalar::<R>(R::secp256k1_n());

        // ---------------------------------------------------------------
        // (1) Bit range: b_i · (b_i − 1) == 0, gated by s_accum.
        //     Expanded to `b · b − b`, both terms gated by s_accum.
        // ---------------------------------------------------------------
        let b1_sq_minus_b1 = b_1.clone() * b_1 - b_1;
        b.assert_zero(s_accum.clone() * &b1_sq_minus_b1);

        let b2_sq_minus_b2 = b_2.clone() * b_2 - b_2;
        b.assert_zero(s_accum.clone() * &b2_sq_minus_b2);

        // ---------------------------------------------------------------
        // (2) Scalar bit accumulation (mod n):
        //     s_accum · (2·U_i + b_i − U_i[t+1] − q_{U_i} · n) == 0
        //     Written as (U + U + b − U^↓1 − q·n) inside parens.
        // ---------------------------------------------------------------
        let q_u1_times_n = mbs(q_u1, &n_scalar).expect("q_U1 · n: arithmetic overflow");
        let accum1_inner =
            u_1.clone() + u_1 + b_1 - down_u_1_sh1 - &q_u1_times_n;
        b.assert_zero(s_accum.clone() * &accum1_inner);

        let q_u2_times_n = mbs(q_u2, &n_scalar).expect("q_U2 · n: arithmetic overflow");
        let accum2_inner =
            u_2.clone() + u_2 + b_2 - down_u_2_sh1 - &q_u2_times_n;
        b.assert_zero(s_accum.clone() * &accum2_inner);

        // ---------------------------------------------------------------
        // (3) Init boundary: U_1[0] == 0, U_2[0] == 0.
        // ---------------------------------------------------------------
        b.assert_zero(s_init.clone() * u_1);
        b.assert_zero(s_init.clone() * u_2);

        // ---------------------------------------------------------------
        // (4) Scalar inverse at row FINAL_ROW:
        //     s_final · (pa_s · w − 1 − q_sw · n) == 0
        //
        //  Expanded so the constant `1` is gated by `s_final` (since
        //  `s_final · 1 = s_final`, we can just subtract `s_final` directly):
        //     s_final · pa_s · w − s_final − s_final · q_sw · n == 0
        // ---------------------------------------------------------------
        let s_times_w = pa_s.clone() * w_inv; // degree 2
        let s_final_sw = s_final.clone() * &s_times_w; // degree 3
        let q_sw_times_n = mbs(q_sw, &n_scalar).expect("q_sw · n: arithmetic overflow");
        let s_final_q_sw_n = s_final.clone() * &q_sw_times_n; // degree 2
        let inv_expr = s_final_sw - s_final - &s_final_q_sw_n;
        b.assert_zero(inv_expr);

        // ---------------------------------------------------------------
        // (5) Signature modular check at row FINAL_ROW:
        //     s_final · k · (k − 1) == 0
        //     s_final · (x_hat − r − k · n) == 0
        // ---------------------------------------------------------------
        let k_sq_minus_k = k.clone() * k - k;
        b.assert_zero(s_final.clone() * &k_sq_minus_k);

        let k_times_n = mbs(k, &n_scalar).expect("k · n: arithmetic overflow");
        let sig_inner = x_hat.clone() - pa_r - &k_times_n;
        b.assert_zero(s_final.clone() * &sig_inner);
    }
}

// ---------------------------------------------------------------------------
// Modular arithmetic helpers (secp256k1 scalar field).
// ---------------------------------------------------------------------------

/// Sample a uniformly-random scalar in `[1, n − 1]`.
fn rand_scalar<Rng: RngCore + ?Sized>(rng: &mut Rng) -> CbUint<ECDSA_INT_LIMBS> {
    let n_nz = NonZero::new(SECP256K1_N_UINT).expect("n is nonzero");
    loop {
        // Sample 320 uniform bits.
        let mut limbs = [0u64; ECDSA_INT_LIMBS];
        for limb in &mut limbs {
            *limb = rng.next_u64();
        }
        // Zero the top limb so the raw value is < 2^256, guaranteeing a
        // bias-free reduction mod n (since n is just below 2^256).
        limbs[ECDSA_INT_LIMBS - 1] = 0;
        let raw = CbUint::<ECDSA_INT_LIMBS>::from_words(limbs);
        let reduced = raw.rem_vartime(&n_nz);
        if !bool::from(reduced.is_zero()) {
            return reduced;
        }
    }
}

/// `(a · b) mod n`, returning the reduced product.
fn mul_mod_n(
    a: &CbUint<ECDSA_INT_LIMBS>,
    b: &CbUint<ECDSA_INT_LIMBS>,
) -> CbUint<ECDSA_INT_LIMBS> {
    // Widen to `Uint<10>` (640 bit) for the raw product, then reduce mod n.
    // `widening_mul` returns (high, low) limbs as a tuple; `.into()` packs
    // them into a single `Uint<ECDSA_INT_LIMBS * 2>`.
    let wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = a.widening_mul(b).into();
    let n_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();
    let n_wide_nz = NonZero::new(n_wide).expect("n is nonzero");
    let (_, rem) = wide.div_rem_vartime(&n_wide_nz);
    rem.resize()
}

/// `a^{-1} mod n` (secp256k1 scalar field; n is odd prime).
fn inv_mod_n(a: &CbUint<ECDSA_INT_LIMBS>) -> CbUint<ECDSA_INT_LIMBS> {
    let n_odd = Odd::new(SECP256K1_N_UINT).expect("n is odd");
    let inv_opt = a.invert_odd_mod(&n_odd);
    inv_opt.expect("a has no inverse mod n (is a a multiple of n?)")
}

// ---------------------------------------------------------------------------
// Int<5> ↔ Uint<5> bridge for trace construction.
// ---------------------------------------------------------------------------

/// Reinterpret a non-negative `Uint<5>` (whose top bit is 0) as an `Int<5>`.
/// All scalars we work with are `< n < 2^256`, so the top bit is always 0
/// and the reinterpretation is the identity as an integer.
fn uint_to_int(u: CbUint<ECDSA_INT_LIMBS>) -> Int<ECDSA_INT_LIMBS> {
    debug_assert!(
        u.bits() <= 256,
        "unexpectedly-large uint: {} bits (should be ≤ 256)",
        u.bits()
    );
    // `as_int` reinterprets the bit pattern. For `u < 2^319` the resulting
    // signed int has the same integer value.
    Int::new(*u.as_int())
}

/// Convert a small `u32` quotient to an `Int<5>`.
fn u32_to_int(q: u32) -> Int<ECDSA_INT_LIMBS> {
    Int::<ECDSA_INT_LIMBS>::from(q)
}

// ---------------------------------------------------------------------------
// Witness generator.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for EcdsaScalarSliceUair<R>
where
    R: EcdsaScalarRing,
    R: From<Int<ECDSA_INT_LIMBS>>,
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
            "ECDSA scalar slice needs > {FINAL_ROW} rows; got {n_rows}"
        );

        // ---- 1. Sample a valid signature (r, s, e). -------------------
        let r = rand_scalar(rng);
        let s = rand_scalar(rng);
        let e = rand_scalar(rng);
        // w = s^{-1} mod n
        let w = inv_mod_n(&s);
        // u_1 = e · w mod n; u_2 = r · w mod n (not constrained in this slice,
        // but we still compute them to drive the bit accumulators honestly).
        let u_1 = mul_mod_n(&e, &w);
        let u_2 = mul_mod_n(&r, &w);

        // ---- 2. Extract 256 bits (big-endian). ------------------------
        let u_1_bits: Vec<u32> = extract_bits_be(&u_1);
        let u_2_bits: Vec<u32> = extract_bits_be(&u_2);
        debug_assert_eq!(u_1_bits.len(), 256);
        debug_assert_eq!(u_2_bits.len(), 256);

        // ---- 3. Build U_i recurrence, track quotient witnesses. -------
        // U_i[0] = 0; U_i[t+1] = (2·U_i[t] + b_i[t]) mod n.
        // In Z: 2·U_i[t] + b_i[t] = U_i[t+1] + q_i[t]·n, q_i[t] ∈ {0, 1}.
        let mut u1_seq: Vec<CbUint<ECDSA_INT_LIMBS>> = Vec::with_capacity(257);
        let mut u2_seq: Vec<CbUint<ECDSA_INT_LIMBS>> = Vec::with_capacity(257);
        u1_seq.push(CbUint::<ECDSA_INT_LIMBS>::ZERO);
        u2_seq.push(CbUint::<ECDSA_INT_LIMBS>::ZERO);
        let mut q_u1_seq: Vec<u32> = Vec::with_capacity(256);
        let mut q_u2_seq: Vec<u32> = Vec::with_capacity(256);
        let n_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();

        for t in 0..256 {
            let step = |prev: &CbUint<ECDSA_INT_LIMBS>, bit: u32|
                -> (CbUint<ECDSA_INT_LIMBS>, u32)
            {
                // Compute 2·prev + bit in Z (widened).
                let prev_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = prev.resize();
                let two_prev = prev_wide.wrapping_shl(1); // 2 · prev
                let bit_wide = CbUint::<{ ECDSA_INT_LIMBS * 2 }>::from_u32(bit);
                let sum = two_prev.wrapping_add(&bit_wide);
                // Quotient and remainder against n.
                let n_nz = NonZero::new(n_wide).expect("n nonzero");
                let (q, rem) = sum.div_rem_vartime(&n_nz);
                let q_limbs = q.to_words();
                let q_u32: u32 = {
                    // Quotient fits in {0, 1}: 2·prev + bit ≤ 2·(n−1) + 1 < 2n.
                    debug_assert!(
                        q_limbs[1..].iter().all(|&l| l == 0) && q_limbs[0] <= 1,
                        "quotient out of {{0,1}}: limbs = {:?}",
                        q_limbs,
                    );
                    q_limbs[0] as u32
                };
                (rem.resize(), q_u32)
            };
            let (u1_next, q_u1_t) = step(&u1_seq[t], u_1_bits[t]);
            let (u2_next, q_u2_t) = step(&u2_seq[t], u_2_bits[t]);
            u1_seq.push(u1_next);
            u2_seq.push(u2_next);
            q_u1_seq.push(q_u1_t);
            q_u2_seq.push(q_u2_t);
        }

        // Sanity: the final accumulator equals u_1 / u_2.
        debug_assert_eq!(u1_seq[256], u_1, "U_1[256] should equal u_1");
        debug_assert_eq!(u2_seq[256], u_2, "U_2[256] should equal u_2");

        // ---- 4. Quotient for scalar inverse: s · w − 1 = q_sw · n. -----
        // Compute q_sw = (s · w − 1) / n in Z (both s · w and 1 treated as
        // unsigned; s · w ≥ 1 always since s, w ∈ [1, n − 1] and their
        // product mod n ≡ 1).
        let sw_wide: CbUint<{ ECDSA_INT_LIMBS * 2 }> = s.widening_mul(&w).into();
        let sw_minus_one = sw_wide.wrapping_sub(&CbUint::<{ ECDSA_INT_LIMBS * 2 }>::ONE);
        let n_nz_wide = NonZero::new(n_wide).expect("n nonzero");
        let (q_sw_wide, rem_sw) = sw_minus_one.div_rem_vartime(&n_nz_wide);
        debug_assert!(
            bool::from(rem_sw.is_zero()),
            "s · w − 1 was not divisible by n (witness inconsistent)"
        );
        let q_sw_uint: CbUint<ECDSA_INT_LIMBS> = q_sw_wide.resize();

        // ---- 5. Pick x̂ and k for signature modular check. --------------
        // Simplest honest witness: k = 0, x̂ = r. Then x̂ − r − k·n = 0.
        let x_hat = r;
        let k_val: u32 = 0;

        // ---- 6. Populate columns. ------------------------------------
        let zero_r = || R::ZERO;
        let mk_col = |_: ()| -> Vec<R> { vec![zero_r(); n_rows] };

        let mut s_init_col = mk_col(());
        let mut s_accum_col = mk_col(());
        let mut s_final_col = mk_col(());
        let mut pa_e_col = mk_col(());
        let mut pa_r_col = mk_col(());
        let mut pa_s_col = mk_col(());
        let mut b1_col = mk_col(());
        let mut b2_col = mk_col(());
        let mut u1_col = mk_col(());
        let mut u2_col = mk_col(());
        let mut w_col = mk_col(());
        let mut xhat_col = mk_col(());
        let mut k_col = mk_col(());
        let mut q_u1_col = mk_col(());
        let mut q_u2_col = mk_col(());
        let mut q_sw_col = mk_col(());

        let one_r = R::ONE;

        // Selectors.
        s_init_col[0] = one_r.clone();
        for t in 0..=255 {
            s_accum_col[t] = one_r.clone();
        }
        s_final_col[FINAL_ROW] = one_r.clone();

        // Public scalars at FINAL_ROW.
        pa_e_col[FINAL_ROW] = R::from(uint_to_int(e));
        pa_r_col[FINAL_ROW] = R::from(uint_to_int(r));
        pa_s_col[FINAL_ROW] = R::from(uint_to_int(s));

        // Bits: rows 0..=255.
        for t in 0..256 {
            b1_col[t] = R::from(u32_to_int(u_1_bits[t]));
            b2_col[t] = R::from(u32_to_int(u_2_bits[t]));
        }

        // Running accumulators: rows 0..=256.
        for t in 0..=256 {
            u1_col[t] = R::from(uint_to_int(u1_seq[t]));
            u2_col[t] = R::from(uint_to_int(u2_seq[t]));
        }

        // Quotient witnesses at each accumulation row.
        for t in 0..256 {
            q_u1_col[t] = R::from(u32_to_int(q_u1_seq[t]));
            q_u2_col[t] = R::from(u32_to_int(q_u2_seq[t]));
        }

        // Row-256 witnesses.
        w_col[FINAL_ROW] = R::from(uint_to_int(w));
        xhat_col[FINAL_ROW] = R::from(uint_to_int(x_hat));
        k_col[FINAL_ROW] = R::from(u32_to_int(k_val));
        q_sw_col[FINAL_ROW] = R::from(uint_to_int(q_sw_uint));

        // ---- 7. Assemble MLEs and UairTrace. --------------------------
        let to_int_mle =
            |col: Vec<R>| -> DenseMultilinearExtension<R> { col.into_iter().collect() };

        let int = vec![
            to_int_mle(s_init_col),
            to_int_mle(s_accum_col),
            to_int_mle(s_final_col),
            to_int_mle(pa_e_col),
            to_int_mle(pa_r_col),
            to_int_mle(pa_s_col),
            to_int_mle(b1_col),
            to_int_mle(b2_col),
            to_int_mle(u1_col),
            to_int_mle(u2_col),
            to_int_mle(w_col),
            to_int_mle(xhat_col),
            to_int_mle(k_col),
            to_int_mle(q_u1_col),
            to_int_mle(q_u2_col),
            to_int_mle(q_sw_col),
        ];

        UairTrace {
            int: int.into(),
            ..Default::default()
        }
    }
}

/// Extract 256 bits of a `Uint<5>` value in big-endian order:
/// `bits[0]` = MSB, `bits[255]` = LSB.
fn extract_bits_be(u: &CbUint<ECDSA_INT_LIMBS>) -> Vec<u32> {
    let words = u.to_words();
    // Only the low 256 bits are meaningful (top limb should be 0).
    debug_assert_eq!(words[ECDSA_INT_LIMBS - 1], 0, "top limb nonzero");
    let mut bits = Vec::with_capacity(256);
    for limb_idx in (0..4).rev() {
        let limb = words[limb_idx];
        for bit_idx in (0..64).rev() {
            bits.push(((limb >> bit_idx) & 1) as u32);
        }
    }
    bits
}

// Correctness is validated by `tests::test_e2e_ecdsa_slice` in
// `protocol/src/lib.rs`, which runs the full prove/verify round-trip.
