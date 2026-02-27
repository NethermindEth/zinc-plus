//! Mathematical specification of all 7 ECDSA verification constraints.
//!
//! These constraints enforce correct ECDSA signature verification over
//! secp256k1 using Shamir's trick with Jacobian coordinates.
//!
//! # Notation
//!
//! - `[t]` is the current row (up), `[t+1]` is the next row (down)
//! - F_p = secp256k1 base field (256-bit prime)
//! - Jacobian coordinates: affine (x, y) ↔ (X : Y : Z) where x = X/Z², y = Y/Z³
//!
//! # Column layout (9 columns)
//!
//! | Index | Name  | Description                                |
//! |-------|-------|--------------------------------------------|
//! | 0     | b₁    | Bit of scalar u₁                           |
//! | 1     | b₂    | Bit of scalar u₂                           |
//! | 2–4   | X,Y,Z | Accumulator point (Jacobian)               |
//! | 5–7   | X_mid, Y_mid, Z_mid | Doubled point (Jacobian)    |
//! | 8     | H     | Addition scratch: chord x-difference       |
//!
//! The scalars u₁ = e·s⁻¹ and u₂ = r·s⁻¹ are public inputs (not in the
//! trace). The quotient bit k is boundary-only. The auxiliary values
//! S = Y² and R_a = T_y·Z_mid³ − Y_mid are inlined as sub-expressions.
//!
//! # Doubling Phase (rows 2–257)
//!
//! Each row computes: 2P → P_mid, then selectively adds G and/or Q
//! based on scalar bits b₁, b₂.
//!
//! ## Constraint 1: Doubled Z-coordinate (F_p)
//!
//! ```text
//! Z_mid[t] = 2·Y[t]·Z[t]   (mod p)
//! ```
//!
//! Z-coordinate of the doubled point using the standard Jacobian
//! doubling formula: Z' = 2·Y·Z.
//!
//! ## Constraint 2: Doubled X-coordinate (F_p)
//!
//! ```text
//! X_mid[t] = 9·X[t]⁴ − 8·X[t]·Y[t]²   (mod p)
//! ```
//!
//! Standard Jacobian doubling formula for the X-coordinate with S = Y²
//! inlined. M = 3·X² (since a=0 for secp256k1), U = 4·X·S = 4·X·Y².
//!
//! ## Constraint 3: Doubled Y-coordinate (F_p)
//!
//! ```text
//! Y_mid[t] = 12·X[t]³·Y[t]² − 3·X[t]²·X_mid[t] − 8·Y[t]⁴   (mod p)
//! ```
//!
//! Standard Jacobian doubling formula for the Y-coordinate with S = Y²
//! inlined.
//!
//! ## Constraint 4: Addition scratch H (F_p)
//!
//! ```text
//! H[t] = T_x · Z_mid² - X_mid   (mod p)
//! ```
//!
//! where T_x is the x-coordinate of the table point selected by (b₁, b₂):
//! - (0,0): identity (no addition)
//! - (1,0): G
//! - (0,1): Q
//! - (1,1): G+Q (precomputed in row 1)
//!
//! H represents the x-coordinate difference for the addition chord.
//!
//! ## Constraint 5: Result Z-coordinate (F_p)
//!
//! ```text
//! Z[t+1] = (1−s)·Z_mid + s·(Z_mid·H)   (mod p)
//! ```
//!
//! where s = b₁ + b₂ − b₁·b₂ (Shamir selector: s=1 iff any bit set).
//!
//! ## Constraint 6: Result X-coordinate (F_p)
//!
//! ```text
//! X[t+1] = (1−s)·X_mid + s·(R_a² − H³ − 2·X_mid·H²)   (mod p)
//! ```
//!
//! where R_a = T_y·Z_mid³ − Y_mid (inlined sub-expression).
//! Standard Jacobian point addition formula for x-coordinate,
//! with Shamir selector for the (0,0) case (identity addition).
//! Max degree 12 due to s(deg2) × R_a²(deg10).
//!
//! ## Constraint 7: Result Y-coordinate (F_p)
//!
//! ```text
//! Y[t+1] = (1−s)·Y_mid + s·(R_a·(X_mid·H² − X[t+1]) − Y_mid·H³)   (mod p)
//! ```
//!
//! where R_a = T_y·Z_mid³ − Y_mid (inlined sub-expression).
//! Standard Jacobian point addition formula for y-coordinate,
//! with Shamir selector for the (0,0) case.
//! Max degree 10 due to s(deg2) × R_a(deg5) × (X_mid·H²)(deg3).
//!
//! # Boundary Constraints
//!
//! In addition to the 7 non-boundary constraints above, the ECDSA
//! verification requires boundary constraints on the first and last rows:
//!
//! - **Row 1**: Initial accumulator = identity point,
//!   Q on curve (Qy² = Qx³ + 7), precompute T = G+Q
//! - **Row 258**: Final check: X_final / Z_final² ≡ r (mod n),
//!   k·n + r = X_final/Z_final² (quotient bit k is boundary-only),
//!   u₁ = e·s⁻¹ (mod n), u₂ = r·s⁻¹ (mod n) (verified from public inputs)
//!
//! # Implementation Status
//!
//! These constraints require the multi-ring UAIR extension because:
//! 1. F_p arithmetic (256-bit Montgomery multiplication) cannot be
//!    expressed in BinaryPoly<32> (which uses F₂ coefficient arithmetic)
//! 2. Different columns use different rings (F_p vs Q)
//! 3. The Shamir selector expressions require multiplying ring elements
//!    by boolean selectors, which needs an appropriate embedding
//!
//! The implementation path follows Option D from the architecture analysis:
//! - Define `impl Uair<Fp256Poly> for EcdsaUair` using 256-bit field polynomials
//! - Run a second IdealCheck + CPR protocol for the F_p constraints
//! - Combine both sub-protocol transcripts in the pipeline
