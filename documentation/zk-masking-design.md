# Zero-knowledge masking for Zip+ — design ledger

Implementation of *"Modular Masking: Zero-Knowledge IOPPs for Codes over the
Rationals"* (companion draft `zk_zip_plus.tex`, June 2026; referred to below as
"the note"). Branch: `main-beta-zk`. Code: `zip-plus/src/pcs/zk/`.

Ledger style: each increment records *what*, *why*, and *state*. Update this
file whenever the design moves.

---

## Increment 1 (2026-06-10): masked commit + blinded open, transparent inner stub

**What.** New module `zip-plus/src/pcs/zk/` with:

- `params.rs` — `ZkMaskParams<WL>`: the note's parameter chain
  (`combo_bits -> blind_bits -> entry_bits -> p_bits`), deterministic
  derivation of the mask prime `p` (smallest probable prime `>= 2^p_bits`,
  Miller–Rabin), and fit checks against `CombR` and the `Int<WL>` remainder
  headroom.
- `mask.rs` — `MaskSeeds<WL>`: rejection-sampled seeds in `[0, p)^D`,
  Vandermonde/Horner mask symbols `[<G_l, s>]_p` over the runtime-modulus
  `MontyField<WL>` (canonical lifts via `retrieve()`), and the blake3 seed
  commitment (the stub for the inner ZK-IOP witness oracle).
- `protocol.rs` — `ZkZip<Zt, Lc, WL>`: `commit_single` / `prove_single` /
  `verify` implementing the note's Protocol 1 for the **scalar lane**
  (`Comb::DEGREE_BOUND == 0`), single polynomial, single opening:
  - commit: `v_i = Enc(w_i) + [G s_i]_p` over `Z` (blinding row 0 via
    `encode_wide`, witness rows via `encode`), Merkle over masked columns,
    salted seed commitment;
  - prove: `b`, then `beta = <w_0, q_1>` (pre-challenge), then coefficients
    (**always sampled, also at `num_rows == 1`** — note §6.1), then
    `w* = w_0 + sum_j rho_j row_j`, column openings, transparent seed reveal;
  - verify: eval check, `w*` range check (`< 2^(B_0+1)`), ZK coherence
    `<w*, q_1> = beta + <coeffs, b>`, per column Merkle + leaf range +
    two-sided remainder range (`|rem| <= R_max = (1 + sum|rho|)(p-1)`;
    two-sided because `Zt::Chal` is signed), seed-commitment re-binding,
    canonical-residue validation, exact lift check (note Eq. (3)).

Tests (14): parameter chain + determinism + fit rejections; mask canonicity,
determinism, Horner consistency, seed-commitment binding; e2e roundtrips at
`num_rows = 4` and `num_rows = 1` (the J = 1 regime); wrong-eval and
tampered-seed-commitment rejection; commitment randomization (hiding smoke);
mask-budget rejection. All 102 `zip-plus` tests pass.

**Decisions.**

- *Single wide limb count `WL`* (tests: `WL = 8`, 512-bit) for masked entries,
  remainder arithmetic, and the mask field; `ZkMaskParams::derive` rejects
  configurations that don't fit. Conversions ride the generic
  `FromRef<Int<L2>> for Int<L>` / `Uint::as_int` plumbing.
- *Mask prime is a public parameter*, derived deterministically from bit
  budgets (never transcript-sampled — masks are committed against it at
  commit time).
- *Mask arithmetic via `MontyField<WL>`* with runtime modulus; canonical
  lifts via `retrieve()`. Mask generation is Horner per position
  (`O(n·D)` big-field muls) — fine at current scale; switch to an
  FFT-friendly `p` or the Galois-ring limb variant (note Rem 5.7) when the
  prover cost matters.
- *Blinding row participates as row 0* with coefficient exactly 1; its
  claimed value `beta` is a separate transcript message written **before**
  the combination challenge (binding order is load-bearing — note Thm 4.2/4.3).
- *Scalar lane only*, enforced both by trait bounds
  (`Int<WL>: FromRef<Zt::Cw>` excludes polynomial codeword types) and a
  runtime `DEGREE_BOUND == 0` check with a pointer to the per-coefficient
  masking plan (note §7 item 3).

**State / known gaps (ordered).**

1. **Inner ZK-IOP missing (the v0 stub reveals seeds).** Binding and complete
   with the final round structure; `w*` is blinded; opened columns are NOT
   hidden yet. Replace `MaskConsistencyProof::Transparent` with the
   `R_lift` ZK-IOP of note §5 (linear over `F_{p'}` + range checks;
   instantiate Ligero-style over a limbed field). This is the seam:
   nothing else changes.
2. **Fiat–Shamir absorption audit.** Like the base PCS, stream messages
   written via `write_const_many` (`w*`, column values, seeds) are *not*
   absorbed into the FS state; `b` and `beta` are (via
   `write_field_elements`). Mirrored deliberately for consistency — audit
   jointly with the base protocol before any production use.
3. **`codeword_growth_bits` is caller-supplied** (test value 80 for the
   depth-1 IPRS code). Derive it from the IPRS norm bound
   (`crypto/crypto_iprs.tex`, Thm `thm:iprs`) and expose it on `LinearCode`.
4. **Not wired into `protocol/`** (`step0/step7`), folded variants, or
   `multi_zip`. Integration changes the commitment semantics to
   (row oracle, seed commitment) — downstream theorem statements and the
   soft bit-bound parameter (`B_0 + 1` instead of `B_1`) must follow
   (note §7 items 1–2).
5. **Soundness of the final scheme** additionally needs affine/coset MCA
   (note Assumption 2.7) — at `num_rows = 1` it reduces to the pairwise
   case closest to what Zinc+ already proves.
6. Polynomial lanes (`Z[X]` coefficients): per-coefficient-slot masks,
   multiplying the inner statement by the degree bound (note §7 item 3).
7. Multi-opening support: provision `mask_dim = kappa * C` and one blinding
   row per revealed combination (note Rem 6.2); currently single-opening.
8. Pre-existing repo issue (not introduced here): `cargo clippy` with the
   current toolchain fails on `main-beta` files outside this module
   (`zinc-poly`, `zip-plus/{utils,folding,multi_zip,iprs,phase_prove}`).
   The `pcs/zk/` module itself is clippy-clean.

**Cost note (J = 1, the common case).** The inner statement collapses to
`2(D + 2C)` values, so the v1 inner ZK-IOP will be negligible there; the
overhead concentrates in `w*` entry growth (`B -> B + B_rho + lambda_zk`) and
the ~`4x` committed-entry growth. See note §6.1 / Table 2.

---

## Increment 2 (2026-06-10): hardening + inner ZK-IOP foundation

**What.**

1. `LinearCode::codeword_growth_bits` (default `None`) with the IPRS
   implementation from the paper's norm bound (`thm:iprs`:
   `(q/2)^(depth+1) * row_len`, rounded up per factor — 41 bits for the
   depth-1 test code) plus an empirical soundness test;
   `ZkMaskParams::derive_for_code` consumes it.
2. FS absorption: the zk opening now absorbs every proof message (`w*`,
   column values, seeds, salt) into the Fiat–Shamir state on both sides, so
   column challenges bind to `w*` and to previous openings. The base PCS's
   `write_const_many` messages are still unabsorbed — that audit item stays
   open for the non-zk path.
3. **Window convention** (prerequisite for the inner constraint system, cf.
   note §5 / power-of-two discussion): `p` is the least prime above
   `2^p_bits`, seeds are sampled from `[0, 2^(bits(p)-1))` (negligibly
   non-uniform mod `p` — prime-gap/p), and commit-time seeds are resampled
   until every mask symbol lands in the same window (probability ~0). Every
   inner range check is therefore a pure bit decomposition; no comparison
   against an arbitrary modulus is ever emitted.
4. **`zk/inner/` foundation:**
   - `inner/field.rs` — deterministic FFT-friendly inner-prime search
     (`p' = c * 2^tau + 1` with its two-adic decomposition tracked), QNR /
     root-of-unity / Fermat-inverse helpers, square-and-multiply `pow`, and
     a radix-2 NTT (forward / inverse / coset evaluation). Tested against
     Horner evaluation and for coset-subgroup disjointness (the Ligero
     message/codeword domain split of the next increment).
   - `inner/constraints.rs` — the `R_lift`-to-bits compiler: `LiftLayout`
     (witness = bits of seeds, lifts `X`, quotients `Y`; index helpers;
     `required_inner_bits` for sizing `p'`), sparse linear constraints over
     `F_p'` (lift equations `<G_l, s_i> = X + p Y` and aggregation
     equations `rem_l = X_0 + sum rho_j X_j` emitted directly over bits —
     Lemma 5.3: linear; booleanity is left to the IOP's quadratic test),
     and the honest-witness builder (exact integer lifts recomputed in
     `Uint<IL>` with `div_rem`). Tests: honest witness satisfies all
     constraints (incl. duplicate columns and a negative weight), tampered
     bit / wrong `rem` violate, bound bookkeeping.

**State.** 23 zk tests, 111 total in `zip-plus`, zk files clippy-clean.
Remaining for v1-inner (next increment): `inner/ligero.rs` — the ZK-Ligero
IOP over `F_p'` (interleaved RS commitment on a coset domain via the NTT
tooling, `t`-symbol random padding for exact column hiding, blinded linear
test + booleanity quadratic test, column spot checks), the split commitment
(seed bits at outer-commit time, `X`/`Y` bits at opening), and the swap of
`MaskConsistencyProof::Transparent`. Open design knob: inner soundness
parameters (spot-check count, padding budget) need calibration; bits-as-
field-elements is wasteful (note Rem 5.6 recommends the limbed field +
lookups eventually) — correctness first.
