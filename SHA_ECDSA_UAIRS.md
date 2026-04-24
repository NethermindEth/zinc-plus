# SHA-256 and ECDSA UAIRs

Two hybrid-arithmetization UAIRs packaged together on this branch as a
minimal reference. Both are *slices* — they cover the scalar/bit parts
of their respective algorithms and mark out what's out of scope.

Source files:

- [`test-uair/src/sha256.rs`](test-uair/src/sha256.rs) —
  `Sha256CompressionSliceUair`, `Sha256Ideal`.
- [`test-uair/src/ecdsa.rs`](test-uair/src/ecdsa.rs) —
  `EcdsaScalarSliceUair`, `EcdsaScalarRing`, `ECDSA_INT_LIMBS`.

## SHA-256 compression slice

Full SHA-256 compression round function + message schedule as a UAIR.
Ties both ends of the compression via public-input boundary rows:

- `init`: `a_hat[0] = y_a_public[0]`.
- `final`: `a_hat[row] = y_a_public[row]` and `e_hat[row] = y_e_public[row]`
  at each of the last four rows, encoding `(d, c, b, a)` / `(h, g, f, e)`
  under the SHA-256 shift-register convention.

**Constraint families** (per active row `t`, see module doc for the full list):

1. `Sigma_0(a)` and `Sigma_1(e)` rotation identities, `F_2[X]` → `Q[X]`-lifted
   via per-coefficient overflow witnesses.
2. `sigma_0(W)` and `sigma_1(W)` rotation identities (same lifting).
3. Right-shift decompositions `W = T_0 + X^3·S_0` and `W = T_1 + X^10·S_1`.
4. Message-schedule modular sum at anchor `t − 16`.
5. Register-update modular sums for `a` and `e` (shift-register trick:
   `h[t] = e[t-3]`, `d[t] = a[t-3]`).
6. Init / final boundary equalities tying the committed registers to
   the public-input columns.

The module doc (`test-uair/src/sha256.rs`) explains the `F_2[X] → Q[X]`
overflow-witness lifting, the soundness caveat around `mod 2`
identities, and what's out of scope (bit-poly / range lookups, `Ch`/`Maj`
enforcement as free witness columns, `K` populated with random
integers instead of SHA-256's canonical round constants).

## ECDSA scalar slice

Minimal ECDSA signature-verification slice covering the scalar-field
(`F_n`) and integer parts only.

**Constraint families** (per module doc):

1. Bit-range: `b_i · (b_i − 1) = 0` for `i ∈ {1, 2}` at rows 0..255.
2. Scalar bit-accumulation mod `n`:
   `2·U_i[t] + b_i[t] − U_i[t+1] − q_{U_i}[t] · n == 0` at rows 0..255.
3. Init boundary: `U_1[0] == 0`, `U_2[0] == 0`.
4. Scalar inverse mod `n`: `s · w − 1 − q_sw · n == 0` at row 256.
5. Signature modular check: `x̂ − r − k · n == 0` with `k · (k − 1) == 0`.

`F_n` identities are lifted to `Z` / `Q[X]` via explicit quotient
witnesses; the prover supplies `q`, the verifier runs the standard
`Q[X]` `assert_zero` check. `q < n` is not range-checked (soundness
gap, deferred to a future lookup-based extension).

**Out of scope** (deferred): all `F_p` / elliptic-curve operation
constraints (Jacobian doubling, addition, Shamir's trick, affine
conversion — needs a wider, degree-6 slice), boundary checks
`u_1 = e·w` and `u_2 = r·w`, and composition with the SHA-256 UAIR
(binding `y_e` to a digest).

## Field parameters

- `Sha256CompressionSliceUair` uses bit-polynomial columns of width 32
  (`BinaryPoly<32>`) and the protocol's default field.
- `EcdsaScalarSliceUair` uses integer columns of 320-bit precision
  (`Int<5>`, where `ECDSA_INT_LIMBS = 5`) to hold secp256k1 scalars with
  enough headroom for the quotient-witness lift.

## Usage

The two UAIRs implement the standard `Uair` and `GenerateRandomTrace`
traits and can be driven by the Zinc+ prover/verifier via
`ZincPlusPiop::prove` / `verify` in the same way as the sample UAIRs
in `test-uair/src/lib.rs`.
