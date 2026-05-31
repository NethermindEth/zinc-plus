# Handoff — F_2 SHA-256 Hadamard discharge

You are taking over the coefficient-wise Hadamard (bitwise-AND) discharge
for the all-`F_2` SHA-256 prover. The per-slice zerocheck AND **the sound
PCS discharge (§4 below) are now implemented, tested (38 protocol lib
tests green), and clippy-clean** on branch `f2-clean` (working tree —
commit it). Your job: extend it to the **real SHA-256 relations**
(shifts / virtual operands, the `W_β` carry column, then register the 16
relations) — see **§5**, which is now the next task.

> **STATUS UPDATE (sound discharge SHIPPED).** §4's "next task" is done.
> `prove_f2_full_with_hadamard` / `verify_f2_full_with_hadamard` exist;
> `F2FullProof` carries the four `hadamard_*` discharge fields; the
> verifier's binding check + the up-eval-absorb make the in-flow
> recombination sound. Full write-up in the ledger
> (`documentation/f2x-sha-todo.md`, entry "SOUND DISCHARGE SHIPPED").
> Read §4 for *what it does*; jump to §5 for *what's next*.

---

## 0. Read these first (in order)

1. **`protocol/src/f2_hadamard_plan.md`** — the design.
   - §1–2: *why* the per-slice / bit-axis approach (the math — read it).
   - **§5.7 / §5.7.1: your immediate next task** (the sound discharge),
     with the exact edit list and the call patterns to copy.
   - §6: dependencies/risks. §7: phases. **Appendix A: the 16 SHA
     Hadamard relations** + which columns they touch.
2. **`documentation/f2x-sha-todo.md`** — the ledger. Search the entry
   "ψ_α-projected sumcheck term for Hadamard discharge": it records the
   design history and **the rejected alternatives with counterexamples**
   so you don't re-walk dead ends. **CLAUDE.md makes updating this ledger
   a hard rule** for any F_2 SHA-256 prover-path work — do it every turn.

---

## 1. Critical math — do NOT regress this

Bitwise AND of two 32-bit words is the **coefficient-wise** product
`W = U ⊙ V` (`W_i = U_i·V_i`). The projection `ψ_α` (evaluate a bit-poly
at `X=α`) is a **ring homomorphism**, so `ψ_α(U)·ψ_α(V) = ψ_α(U·V)` where
`·` is the **polynomial/convolution** product — *not* the AND.
Counterexample: `U=V=1+X` ⇒ `ψ(U)ψ(V)=(1+α)²=1+α²`, but
`ψ(U⊙V)=ψ(1+X)=1+α`. So you **cannot** discharge AND with a single
`ψ(U)·ψ(V)−ψ(W)` sumcheck term — it proves the wrong relation and is
unsound. Two `(U,V)` pairs can share a convolution but differ in AND
(`1+X, 1+X³` vs `1+X², 1+X+X²` — both have product `1+X+X³+X⁴`, ANDs `1`
vs `1+X²`), so no convolution-domain trick recovers it. The implemented
fix keeps the bit index a **live sumcheck variable** (a per-coefficient
zerocheck, exactly like the booleanity sumcheck). If you're tempted to
"simplify" back to a ψ-product, re-read the ledger first.

---

## 2. What's committed and working (`a110d32` → `cddd5ed`)

- **`piop/src/lookup/hadamard.rs`** — the per-slice cross-product
  zerocheck: `prepare_hadamard_group` / `finalize_hadamard_{prover,
  verifier}` / `prepare_hadamard_verifier` + `HadamardTriple`. It mirrors
  `piop/src/lookup/booleanity.rs` with the self-product `v(v−1)` replaced
  by the two-column `U·V − W`. Unit-tested.
- **`protocol/src/f2_hadamard.rs`** — protocol phase helpers:
  `prove_f2_hadamard_phase` / `verify_f2_hadamard_phase`, `F2HadamardSpec`
  (`{u_col,v_col,w_col}`), `F2HadamardProof`, and `alpha_parent_evals`.
  Tested over real `BinaryPoly` columns including the recombination.
- **`protocol/src/f2_prove.rs`** — the Hadamard zerocheck is threaded into
  `prove_f2_uair_with_groups` / `verify_f2_uair_with_groups` (**Wiring R**:
  it runs *before* α, reusing the IC point `r`; α is then drawn fresh and
  doubles as the recombination element). Added: `F2Proof.{hadamard_proof,
  hadamard_parent_evals}`; `F2VerifierSubclaim.{hadamard_rstar,
  hadamard_distinct}` (exposed for the sound discharge); `F2VerifyError`
  +4 variants; the e2e test `hadamard_phase_roundtrips_in_flow`.

Commit trail:
```
cddd5ed expose r*_H + distinct cols (sound-discharge plumbing)
f44cdf3 docs: concrete sound-discharge design (§5.7.1)
13c169e per-slice Hadamard discharge wired into F_2 prove/verify (trusted)
c0f92b1 WIP: GF128 refactor (the branch owner's, landed cleanly)
953a918 docs: sound-discharge spec
390c846 alpha_parent_evals helper
ade7ab9 Wiring-R phase helpers (A1 core)
a110d32 per-slice cross-product zerocheck (A0)
```
Working tree is clean; all 37 `zinc-protocol` lib tests pass.

---

## 3. Soundness posture — NOW SOUND (gap closed)

The in-flow recombination check
`Σ_b α^b·v_b(r*_H) == parent_eval` (`verify_bit_decomposition_consistency`)
ties the per-slice evals to `parent_evals`. `parent_evals` **used to be**
prover-supplied / trusted (honest-prover-only). They are now bound to the
commitment by the second mp+open at `r*_H` plus the verifier's binding
check `hadamard_evals_at_rstar_h[distinct] == parent_evals` (§4). The
recombination is therefore **sound**. (The remaining honest-prover-only
piece is *row-shift* discharge for shifted operands — Phase B/C, §5 +
ledger Issue 1 — not the AND discharge itself.)

---

## 4. DONE — sound discharge (Approach A). Full detail in §5.7.1.

`parent_evals` are now opened at `r*_H` against the commitment. What
shipped (all in `protocol/src/f2_prove.rs`):

1. **`F2FullProof`** gained FOUR fields (all `None`/empty without
   Hadamard): `hadamard_multipoint_eval: Option<MultipointEvalProof>`,
   **`hadamard_evals_at_rstar_h: Vec<Gf>`** (every projected column's eval
   at `r*_H` — the second mp's `up_evals`; this one is *not* in the
   original §5.7.1 list but is **required** — the verifier needs the
   up-evals and they get bound transitively),
   `hadamard_open_evals_at_r0h: Vec<Gf>`, `hadamard_open: Option<F2OpenProof>`.
2. **`prove_f2_full_with_hadamard` / `verify_f2_full_with_hadamard`** added;
   existing `*_with_bit_ops` (+ pre-paired) signatures **unchanged** (benches
   untouched). Bodies factored into private `prove_f2_full_impl` /
   `verify_f2_full_impl(.., hadamard_specs)` (old entries pass `&[]`).
3. **Second mp+open** after the main one: evaluate all projected cols at
   `r*_H`, **absorb those up-evals** (soundness-critical — see below), run a
   second `MultipointEval` → `r_0^H`, then `prove_f2_open` on the same
   witness slice at `r_0^H`.
4. **Verify**: mirror, then binding check
   `hadamard_evals_at_rstar_h[distinct] == proof.uair.hadamard_parent_evals`.
   The in-flow recombination is kept as-is; the binding makes it sound.
5. **e2e test** `prove_then_verify_f2_full_with_hadamard_roundtrips`
   (HadF2Uair): honest round-trip + 4 tamper rejections.

⚠️ **Soundness detail you must preserve if you touch this**: the second
mp's `up_evals` (`hadamard_evals_at_rstar_h`) are absorbed into the
transcript **before** the second mp samples γ (both sides), mirroring how
`prove_f2_uair_with_groups` absorbs `column_evals_at_rstar` before the main
mp. Without that absorb the per-column up-evals aren't SZ-pinned and the
binding is defeatable. Don't drop it.

Optimisation left open (ledger): collapse only the Hadamard subset, or
fold `r*`+`r*_H` into one two-point multipoint-eval (the "proper" mp the
ledger's **Issue 1** also needs) — one open instead of two.

---

## 5. NEXT TASK — real SHA-256 relations (shifts, `W_β`, register the 16)

The current tests use a synthetic shift-free `HadF2Uair`. The real
SHA-256 relations (Appendix A) need:
- **Shifts / virtual operands**: `↓Δ` row-shifts, `X·` bit-shifts, `1−`,
  `A+A^↓2`. Bit-shifts and `F_2`-linear combos fold into the bit-slice
  MLEs (reuse `build_virtual_booleanity_mles` / the bit-axis). **Row
  shifts need a row-shift discharge** — MLE evaluation does *not* commute
  with row shifts; this is shared with the ledger's **Issue 1**
  (currently the multipoint-eval is degenerate/single-point).
  Honest-prover-first is acceptable to get e2e first.
- **`W_β` carry column** (`f2_hadamard_plan.md` §4.5): required for the 13
  adder relations; the 3 AND relations (C12/C13/C14) work without it. Add
  it to `test-uair/src/sha256_f2.rs` (mirror the `PA_C` setup) + the
  missing `ShiftSpec`s.
- Then register the 16 relations (Appendix A) and run e2e on `sha256_f2`.

---

## 6. Build / test (read this — non-obvious)

- **The `zinc-protocol` crate only compiles with `--features parallel`**
  (the F_2 round-1 fast path uses rayon `reduce`). Plain
  `cargo test -p zinc-protocol --lib` FAILS to build; always pass the
  feature.
- Hadamard tests: `cargo test -p zinc-protocol --features parallel --lib hadamard`
  (3 tests). Full: `… --features parallel --lib` (37 tests, ~40s).
- piop: `cargo test -p zinc-piop --lib hadamard`.
- Clippy is strict (denies `arithmetic_side_effects`, `unwrap_used`,
  lossy casts, …). Put `#[allow(clippy::arithmetic_side_effects)]` on hot
  fns (see `hadamard.rs`), use `.expect(...)` not `.unwrap()`; `#[cfg(test)]`
  code is exempt.

---

## 7. Landmines

- `f2_prove.rs` is **no longer entangled** — the branch owner's GF128 WIP
  landed at `c0f92b1`, so commit normally now.
- `build_shifted_bit_slice_mles` **asserts shift ≠ 0** (`uair/src/lib.rs:259`).
  For Δ=0 (the column itself) use `compute_bit_slices_flat`. The shift
  builder is for the shifted operands in the later phase.
- **Transcript order must match prover/verifier exactly** (Wiring R: the
  Hadamard sumcheck runs before α; `parent_evals` are absorbed *after* α).
  `prepare_hadamard_group` samples `γ'` then `σ`; the verifier must sample
  in the same order.
- `verify_f2_hadamard_phase` returns `(distinct, r*_H)`; the recombination
  uses `verify_bit_decomposition_consistency` from
  `piop/src/lookup/booleanity.rs`.
- Soundness of the recombination element: α must be fresh w.r.t. the
  bit-slice evals — Wiring R guarantees this (α sampled after the Hadamard
  sumcheck). Don't reorder it.
- Update `documentation/f2x-sha-todo.md` whenever you ship/try/reject
  anything on this path (CLAUDE.md hard rule).
