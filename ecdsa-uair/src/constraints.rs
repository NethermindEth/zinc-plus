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
//! # Column layout (11 columns)
//!
//! | Index | Name  | Description                                |
//! |-------|-------|--------------------------------------------|
//! | 0     | b₁    | Bit of scalar u₁                           |
//! | 1     | b₂    | Bit of scalar u₂                           |
//! | 2–4   | X,Y,Z | Accumulator point (Jacobian)               |
//! | 5–7   | X_mid, Y_mid, Z_mid | Doubled point (Jacobian)    |
//! | 8     | H     | Addition scratch: chord x-difference       |
//! | 9     | sel_init  | Selector: 1 at row 0 (public)          |
//! | 10    | sel_final | Selector: 1 at row 257 (public)        |
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
//! # Boundary Constraints (B3–B4)
//!
//! In addition to the 7 non-boundary constraints above, the ECDSA
//! verification requires 2 boundary constraints using selector columns:
//!
//! ## Booleanity of b₁, b₂ (verified outside the constraint system)
//!
//! The scalar-bit columns b₁ and b₂ must be in {0, 1} at every row.
//! Because these columns are **public**, the verifier checks booleanity
//! directly on the raw column data — an O(N) scan of values — rather
//! than imposing two degree-2 algebraic constraints.  This saves
//! constraint-system cost without weakening soundness.
//!
//! ## Constraint B3: Initialization (selector-gated)
//!
//! ```text
//! sel_init · Z = 0
//! ```
//!
//! At row 0 (`sel_init = 1`) this forces Z = 0, meaning the initial
//! accumulator is the Jacobian identity point. At all other rows
//! `sel_init = 0` makes the constraint vacuous.
//!
//! ## Constraint B4: Final signature check (guarded)
//!
//! ```text
//! sel_final · Z · (X − R_SIG · Z²) = 0
//! ```
//!
//! At the final row (`sel_final = 1`), when Z ≠ 0 this enforces
//! X/Z² = R_SIG (i.e. the affine x-coordinate equals k·n + r).
//! When Z = 0 (identity / point at infinity) the constraint is
//! vacuously satisfied — a separate non-degeneracy check (Z ≠ 0)
//! would be needed in a production system.
//!
//! # Column layout (11 columns)
//!
//! | Index | Name       | Description                                 |
//! |-------|------------|---------------------------------------------|
//! | 0     | b₁         | Bit of scalar u₁                            |
//! | 1     | b₂         | Bit of scalar u₂                            |
//! | 2–4   | X,Y,Z      | Accumulator point (Jacobian)                |
//! | 5–7   | X_mid, Y_mid, Z_mid | Doubled point (Jacobian)         |
//! | 8     | H          | Addition scratch: chord x-difference        |
//! | 9     | sel_init   | Selector: 1 at row 0, 0 elsewhere (public)  |
//! | 10    | sel_final  | Selector: 1 at row 257, 0 elsewhere (public)|
//!
//! # Implementation Status
//!
//! The non-boundary constraints (C1–C7) and boundary constraints (B3–B4)
//! are fully implemented for `DensePolynomial<i64, 1>` and `Int<4>` rings.
//! Booleanity of b₁/b₂ is enforced by the verifier on public column data.
//! The `BinaryPoly<32>` variant remains a placeholder (0 constraints)
//! because F_p arithmetic cannot be expressed in F₂[X].
//!
//! The constraints use plain integer arithmetic (no modular reduction),
//! which works for the toy curve (F₁₀₁) but would require 256-bit
//! Montgomery field arithmetic for real secp256k1.
