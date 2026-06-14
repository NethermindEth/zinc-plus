# Phase 3 design note — does the dual ψ_α/ψ_z read-off survive a Basefold/WHIR code-switch?

Branch context: `f2-clean` (SHA-256 all-`F_2[X]` path) × `zip-plus-basefold`
(integer limbwise-WHIR IOPP). This note is the **paper-validation step** of the
proof-size plan recorded in `documentation/f2x-sha-todo.md`
("Code-switching opening (Blaze-style) via the Zip++ binary basefold lane").
No code yet — this settles whether the approach is sound before Phase 1/2 build.

## Verdict (TL;DR)

**Yes, the dual read-off survives — unconditionally on the *fold mechanism*, and
conditionally on one clean, self-enforcing invariant.** The bit-slice fold `a'`
and the two projections (`ψ_α`, `ψ_z`) are **decoupled** from the proximity
machinery: in the current open they are checks (2) (projection discharge) +
the `ψ_z` binding, which are *local linear functionals of `a'`* and never touch
the codeword. A Basefold/WHIR opening replaces only checks (1),(3),(4) — the
"`a'` is the genuine `eq(r_0)`-fold + proximity" machinery. So the read-off
survives **iff the IOPP certifies `a'` as the un-collapsed width-`w` object**
`a' ∈ K[X]_{<w}` (not a scalar `ψ_ξ(a')` for one `ξ`), with `ψ_α`/`ψ_z` applied
*after* the IOPP exactly as today.

The single invariant to enforce:

> **The cell axis `X` (the `w=32` bit coefficients) is passive shared width:
> never folded, never batched, never projected inside the recursion. All fold
> challenges, query positions, and twiddles are `X`-independent `K`-scalars.**

This is the recursive extension of the discipline the one-shot open already
follows (commitment.tex Rem. "Why bit-sliced", lines 73–84: keep eq-weights in
`K`, fold the `{0,1}` slices, *retain all bit slices rather than a single
collapsed value*; realized in code as the un-lifted `GF128[X]<D>` open, commit
`7827683`). The obvious implementation satisfies the invariant automatically;
you have to go out of your way to break it. The fallback if a future variant
*does* break it (run the IOPP twice, once per functional) is in §9.

## 1. The question

The current bit-sliced open binds one object per column — the bit-slice fold —
and reads **both** projections off it (commitment.tex §4.3, f2_prove.rs:3194-3199):
```
a'_g = Σ_b a'_{g,b} X^b ∈ K[X]_{<w},   a'_{g,b} = Σ_x eq_x(r_0)·(w_g)_b(x) ∈ K
ψ_α(a'_g) = Σ_b a'_{g,b}·α^b   = MLE_{ψ_α(w_g)}(r_0)      [linear part]
ψ_z(a'_g) = Σ_b a'_{g,b}·L_b(z) = MLE_{ψ_z(w_g)}(r_0)     [Hadamard discharge]
```
This dual read-off is what makes the AND/Hadamard constraints cost **no second
opening**. Replacing the Brakedown opening with a recursive Basefold/WHIR IOPP
(the Zip++ binary lane) must preserve it, or the whole value proposition of the
all-`F_2[X]` arithmetization is lost. Phase 3 = decide if it does.

## 2. What the current open actually does (the decoupling)

Per commitment.tex §4.4, the open runs four checks on the γ-batched
`a' = Σ_g γ_g a'_g`:

| # | Check | Touches the codeword? | Role |
|---|-------|----------------------|------|
| 1 | Eval consistency: `a'` is the `eq(·;r_0)` fold of the row-fold vector | indirectly (via b') | bind `a'` to data |
| 2 | **Projection discharge**: `ψ_ξ(a') = `γ-batched claims (ξ=α; ξ=ψ_z weights via the binding) | **no** | **the dual read-off** |
| 3 | Coherence: combined row ↔ row-fold vector | indirectly | bind `a'` to data |
| 4 | Encoding/proximity: `t=987` column openings encode the combined row | **yes** | proximity, dominates cost |

Check (2) and the `ψ_z` binding (`ψ_z(a') = Σ_g γ_g·z_r0_evals[g]`,
f2_prove.rs:1327-1337) are **pure `K[X]_{<w}` arithmetic on `a'`** — length-`w`
inner products against fixed weight vectors `α^b` and `L_b(z)`. They are blind
to *how* `a'` was certified. Checks (1),(3),(4) are the only ones that bind `a'`
to the committed columns. **This is the entire structural lever: the read-off is
a post-processing of `a'`, orthogonal to the proximity argument.**

## 3. What Basefold/WHIR does

Basefold reduces a multilinear-evaluation claim `w̃(r_0) = e_0` to a tiny tail by
folding the `ν` row-variables, one per round (Zip++ `iopp.rs`):
```
round r:  half-claims (e^e, e^o);  verifier checks (1−z_r)e^e + z_r e^o = e_{r-1}
          fold w_r = α^e·w^e + α^o·w^o (fresh K challenges);  new claim e_r
          cross-round consistency at spot positions via the (lifted) butterfly
```
The accumulated final claim is exactly `Σ_x eq_x(r_0)·w(x) = w̃(r_0)`, and the
spot-checks certify proximity. **In the integer/standard lane the claim `e_r` is
a scalar in `F`** (Zip++ `half_claims: Vec<Vec<(F,F)>>`, `F: PrimeField`). So a
naive port would certify a single scalar — i.e. `ψ_ξ(a')` for **one** `ξ`,
collapsing the bit slices and killing the other projection. That is the only real
risk, and §4 shows it is avoidable.

## 4. Core argument — orthogonality of the two axes

There are two axes:
- **fold axis** = the `ν` row-variables `{0,1}^ν`, consumed by the IOPP rounds
  (weighted by `eq(r_0)`);
- **width axis** = the `w=32` cell bits (the variable `X`), indexing the
  coefficients of `K[X]_{<w}`.

Every operation in the pipeline is **`K`-linear and acts coordinate-wise on the
width axis**:
- the `eq(r_0)` fold (eq-weights are `K`-scalars, applied to each bit slice
  independently — commitment.tex line 56);
- the `F_2`-RAA code and any foldable inner code (`F_2`-/`K`-linear, "extends
  scalar-wise to `K`-coefficient rows" — commitment.tex lines 41-43);
- the Basefold round identity `(1−z_r)e^e + z_r e^o = e_{r-1}` (`z_r ∈ K` scalar);
- the fold `α^e·w^e + α^o·w^o` (`α ∈ K` scalars);
- the butterfly consistency `c_j = A + τ_j B` (`τ_j ∈ K` twiddle).

Therefore, if the IOPP carries **`K[X]_{<w}`-valued** oracles/claims (equivalently:
`w` parallel scalar Basefold instances sharing *the same* challenges, query
positions, and twiddles), every round identity holds **coordinate-wise in `X`**,
and the final certified claim is the full width-`w` vector
```
e_R = (a'_0, …, a'_{w-1}) = a' ∈ K[X]_{<w},   a'_b = w̃_b(r_0)
```
where `w_b` are the `w` bit-slice MLEs. This is identical to the object the
current open binds — Basefold just computes the `eq(r_0)` fold *recursively*
instead of in one matrix pass, and subsumes checks (1),(3),(4). Check (2) and the
`ψ_z` binding are then applied to `e_R = a'` **unchanged**.

No width factor is *added* relative to today: the current open already carries
width-`w` (`combined_row'`, `b'` are `GF128[X]<D>` = degree-`<32`). Basefold keeps
the same width while recursing the row fold.

**Conclusion: the dual read-off survives.** It is a property of the *folded
object*, and the fold preserves the width axis by linearity.

## 5. The invariant, and the four places to enforce it

A collapse of the width axis can only enter through a step that mixes the `w`
coefficients. Enumerated exhaustively:

1. **Column γ-batching** `a' = Σ_g γ_g a'_g`. Batches over the `K` *columns* `g`,
   not over `X`. ✓ already X-preserving (unchanged from today).
2. **Per-round fold challenges** `α^e, α^o` and **eval coords** `z_r`. Must be
   `K`-**scalars** broadcast across all `w` coordinates — *not* `K[X]_{<w}`
   elements (a poly challenge would convolve the bit slices). ✓ enforce: challenge
   type is `K`, not `K[X]`.
3. **Code-switch / proximity batching** (if a random linear combination is used to
   reduce many proximity claims to one). Must batch over rows/columns, **never over
   `X`**. ✓ enforce: the batching vector indexes rows/cols, width rides along.
4. **The tail.** Sent/checked as a `K[X]_{<w}` (width-`w`) object, not projected to
   a scalar. ✓ enforce: tail type is `K[X]_{<w}`; projection happens only in the
   post-IOPP check (2).

All four are "type-level" obligations on the Phase-1 lane (see §7). They are the
recursive image of the one-shot open's "un-lifted" discipline.

**Reuse note.** We already operate a "fold the row axis, keep `X` as a separate
oblong axis" pattern: the **oblong Hadamard discharge** (`f2_oblong_hadamard.rs`,
`poly/src/univariate/oblong_and.rs`) does exactly this for the zerocheck (Phase-1
bit-skip over `X` + Phase-2 row sumcheck). The width-`w` Basefold is the same axis
separation applied to the *proximity IOPP* instead of the *zerocheck*. The
subspace-Lagrange basis `base_lagrange_at(z)` used for `ψ_z` is the same primitive
the binary foldable chain (Phase 1a) will build on.

## 6. Design refinement — code-switch vs. direct foldable commit

Blaze keeps the fast RAA commit and **code-switches** to a foldable RS code for
the open (RAA is not FRI-foldable). But Phase 3's verdict is **independent of this
choice** — it depends only on the fold preserving `X`. That opens a simpler option:

- **(3a) RAA commit + code-switch** (faithful Blaze): preserves the RAA encode;
  needs code-switching, which adds soundness term §8(c) **and** is the one extra
  place (item 3 of §5) where an `X`-collapse could be introduced.
- **(3b) Commit directly with the foldable IPRS chain, no code-switch**: Basefold
  folds the commitment code directly; item 3 of §5 disappears entirely. The cost is
  a different commit code — but the ledger shows **encode is only ~1.2 ms and
  negligible; commit is ~90% Merkle** (f2x-sha-todo.md, "Commit is the ONLY phase
  Binius64 beats us on"). So giving up RAA's XOR encode for a foldable chain is
  **likely free in practice**, and it removes a soundness term and a collapse risk.

**Recommendation:** evaluate (3b) first. If the foldable IPRS chain's encode +
Merkle stays within a few ms of the RAA path (very likely, since both are
linear-time and Merkle dominates), prefer (3b) — it is strictly simpler to make
sound and to keep `X`-passive. Keep (3a) as the fallback if the chain encode is
unexpectedly costly at our shapes. *(This is a Phase-1 measurement, cheap to run.)*

## 7. What the Phase-1 binary IOPP must expose (API obligations)

To keep §4–§5 true, the binary lane must:

- **Oracle / leaf value type** = `K[X]_{<w}` (width-`w`), i.e. `w` `K`-coefficients
  per position — *not* a scalar `K`. (Today's Zip++ lane is `F`-scalar; this is the
  one substantive generalization. Limbs are gone — over `K` folding doesn't widen,
  so `k=1`, no balanced base-2^p decomposition, no mod-q0 projection.)
- **Claim type** (`half_claims`, tail) = `K[X]_{<w}`.
- **Challenge / eval-coord / twiddle types** = `K` scalars, broadcast over width.
- **Final claim** returned to the host as `a' ∈ K[X]_{<w}` (per γ-batched column),
  on which the host runs the *unchanged* check (2) + `ψ_z` binding.
- **`size_bytes`** accounting on the compiled proof, to confirm the ~150–250 KB
  target at our shapes before Phase 2 wiring (the §0 checkpoint).

Everything else (Merkle pair-bundled leaves, FS schedule, `Q` spot-checks,
`compiled.rs`) is reused as-is — width-`w` leaves change only the leaf serialization,
not the round/query logic.

## 8. Soundness sketch (for Phase 4)

The opening error decomposes cleanly; only the proximity term changes shape:
- (a) **Proximity**: `ε_prox(t,δ)` (Brakedown, `t=987` direct openings) → the
  Basefold/WHIR round-wise proximity error `Σ_rounds (1−δ_r)^{Q}` (or the WHIR
  proximity-gap bound). This *replaces* check (4).
- (b) **Read-off** (check 2 + `ψ_z` binding): unchanged, `O((w−1)/|K|)` per family
  (Schwartz–Zippel on the length-`w` inner product) — the width-`w` claim is
  certified exactly, so these terms are identical to today.
- (c) **Code-switch** (only under 3a): the code-switching reduction's soundness
  term; must be stated to *batch over rows/cols, not `X`* (§5 item 3). Absent under
  3b.
- (d) **`ψ_z` operand binding gap** (adder carries trusted; hadamard.tex Rem.,
  f2_prove.rs:915-916): **orthogonal** — it concerns whether the `ψ_z` parents are
  honestly supplied, not the opening mechanism. Code-switching neither helps nor
  hurts it; it stays the one recorded follow-up, exactly as in the current scheme.

So `ε_open ≤ ε_prox^{WHIR} + O(1/|K|) (+ ε_switch under 3a)`, structurally the same
bound as commitment.tex (110) with the proximity term swapped.

## 9. Fallback (if a future variant breaks the invariant)

If some downstream optimization forces an `X`-collapse inside the recursion (e.g. a
challenge that must be a `K[X]` element), the read-off can still be recovered by
running the IOPP **twice** — once with the claim pre-projected through `ψ_α`, once
through `ψ_z`. Cost: a second opening (≈2× the IOPP proof + a second proximity
pass), forfeiting the "AND for free" sharing. This is strictly worse than the
width-`w` single-open and should be a last resort, but it still beats the current
1.1 MiB (two ~200 KB opens < 1.1 MiB) and it is a clean degradation, not a
correctness failure. **Not expected to be needed** — the natural width-`w`
implementation keeps the invariant.

## 10. Residual questions for Phase 4 (analysis, post-prototype)

- Exact `ε_prox^{WHIR}` at our `(ν, w, rate, Q)` and how `Q` trades against proof
  size — the real driver of the final KB number (the §0 checkpoint measures it).
- (3a only) which code-switching variant (Rothblum code-switching for IOPs vs.
  Diamond–Posen FRI-Binius style) gives the cleanest `X`-passive statement.
- Whether the `eq(r_0)` accumulation in the round identities can also absorb the
  `q_1'` "coherence" half so that *one* fold subsumes checks (1)+(3) with no
  residual `K[X]_{<w}` identity — almost certainly yes (it is just the second half
  of the eq tensor), to confirm during Phase 2.

---

**Bottom line for the plan:** Phase 3 clears. The dual read-off is not endangered
by code-switching; it is a post-IOPP read on the width-`w` folded object, and the
fold preserves width by linearity. The Phase-1 lane's only substantive new
requirement is `K[X]_{<w}`-valued (not scalar) oracles/claims, with `K`-scalar
challenges — and the integer lane's hardest machinery (limbs) *drops out*. Proceed
to Phase 1, and use its checkpoint to (i) confirm the KB target and (ii) decide
3a vs 3b by measuring the foldable-chain commit cost.
