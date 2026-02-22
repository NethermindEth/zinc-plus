//! Mathematical specification of all 11 ECDSA verification constraints.
//!
//! These constraints enforce correct ECDSA signature verification over
//! secp256k1 using Shamir's trick with Jacobian coordinates.
//!
//! # Notation
//!
//! - `[t]` is the current row (up), `[t+1]` is the next row (down)
//! - F_p = secp256k1 base field (256-bit prime)
//! - F_n = secp256k1 scalar field (group order)
//! - Jacobian coordinates: affine (x, y) ↔ (X : Y : Z) where x = X/Z², y = Y/Z³
//!
//! # Doubling Phase (rows 2–257)
//!
//! Each row computes: 2P → P_mid, then selectively adds G and/or Q
//! based on scalar bits b₁, b₂.
//!
//! ## Constraint 1: Scalar accumulation u₁ (F_n)
//!
//! ```text
//! u₁[t+1] = 2·u₁[t] + b₁[t]   (mod n)
//! ```
//!
//! Enforces that `u₁` accumulates the binary representation of the scalar.
//! After 256 rows: `u₁ = Σᵢ b₁[i]·2^(256-i)`.
//!
//! ## Constraint 2: Scalar accumulation u₂ (F_n)
//!
//! ```text
//! u₂[t+1] = 2·u₂[t] + b₂[t]   (mod n)
//! ```
//!
//! Same as C1 for the second scalar.
//!
//! ## Constraint 3: Doubling scratch S (F_p)
//!
//! ```text
//! S[t] = Y[t]²   (mod p)
//! ```
//!
//! Auxiliary variable for the point doubling formula.
//!
//! ## Constraint 4: Doubled Z-coordinate (F_p)
//!
//! ```text
//! Z_mid[t] = 2·Y[t]·Z[t]   (mod p)
//! ```
//!
//! Z-coordinate of the doubled point using the standard Jacobian
//! doubling formula: Z' = 2·Y·Z.
//!
//! ## Constraint 5: Doubled X-coordinate (F_p)
//!
//! ```text
//! X_mid[t] = M² - 2·U   (mod p)
//! ```
//!
//! where M = 3·X² (since a=0 for secp256k1) and U = 4·X·S.
//! Expanding: X_mid = 9·X⁴ - 8·X·Y².
//!
//! Standard Jacobian doubling formula for the X-coordinate.
//!
//! ## Constraint 6: Doubled Y-coordinate (F_p)
//!
//! ```text
//! Y_mid[t] = M·(U - X_mid) - 8·S²   (mod p)
//! ```
//!
//! where M = 3·X², U = 4·X·S, S = Y².
//! Standard Jacobian doubling formula for the Y-coordinate.
//!
//! ## Constraint 7: Addition scratch H (F_p)
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
//! ## Constraint 8: Addition scratch R_a (F_p)
//!
//! ```text
//! R_a[t] = T_y · Z_mid³ - Y_mid   (mod p)
//! ```
//!
//! where T_y is the y-coordinate of the selected table point.
//! R_a represents the y-coordinate difference for the addition chord.
//!
//! ## Constraint 9: Result Z-coordinate (F_p)
//!
//! ```text
//! Z[t+1] = selector(b₁, b₂, Z_mid, H)   (mod p)
//! ```
//!
//! The Z-coordinate of the addition result:
//! - If (b₁,b₂) = (0,0): Z[t+1] = Z_mid (no addition, just doubling)
//! - Otherwise: Z[t+1] = Z_mid · H (standard addition formula)
//!
//! Expressed as: `Z[t+1] · (1 - b₁·(1-b₂) - (1-b₁)·b₂ - b₁·b₂)
//!               + (Z_mid · H) · (b₁·(1-b₂) + (1-b₁)·b₂ + b₁·b₂) = 0`
//!
//! ## Constraint 10: Result X-coordinate (F_p)
//!
//! ```text
//! X[t+1] = R_a² · H⁻² - H³ - 2·X_mid·H⁻²   (mod p)
//! ```
//!
//! Standard Jacobian point addition formula for x-coordinate,
//! with Shamir selector for the (0,0) case (identity addition).
//!
//! ## Constraint 11: Result Y-coordinate (F_p)
//!
//! ```text
//! Y[t+1] = R_a · (X_mid·H⁻² - X[t+1]) - Y_mid·H⁻³   (mod p)
//! ```
//!
//! Standard Jacobian point addition formula for y-coordinate,
//! with Shamir selector for the (0,0) case.
//!
//! # Boundary Constraints
//!
//! In addition to the 11 non-boundary constraints above, the ECDSA
//! verification requires boundary constraints on the first and last rows:
//!
//! - **Row 1**: Initial accumulator = identity point, u₁[1] = u₂[1] = 0,
//!   Q on curve (Qy² = Qx³ + 7), precompute T = G+Q
//! - **Row 258**: Final check: X_final / Z_final² ≡ r (mod n),
//!   u₁ = e·s⁻¹ (mod n), u₂ = r·s⁻¹ (mod n)
//!
//! # Implementation Status
//!
//! These constraints require the multi-ring UAIR extension because:
//! 1. F_p arithmetic (256-bit Montgomery multiplication) cannot be
//!    expressed in BinaryPoly<32> (which uses F₂ coefficient arithmetic)
//! 2. Different columns use different rings (F_p vs F_n vs Q)
//! 3. The Shamir selector expressions require multiplying ring elements
//!    by boolean selectors, which needs an appropriate embedding
//!
//! The implementation path follows Option D from the architecture analysis:
//! - Define `impl Uair<Fp256Poly> for EcdsaUair` using 256-bit field polynomials
//! - Run a second IdealCheck + CPR protocol for the F_p constraints
//! - Combine both sub-protocol transcripts in the pipeline
