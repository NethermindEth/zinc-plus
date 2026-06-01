# Design pass: univariate-skip / small-characteristic discharge (the Binius scaling lever)

**Status:** design only, no code. Decision artifact for whether to build the
Binius-style univariate-skip Hadamard discharge. Cross-ref:
`documentation/f2x-sha-todo.md` ("Univariate skip … the Binius scaling lever").

## 1. The problem, precisely

The F₂ SHA-256 Hadamard discharge is **~92 % of the nvars=16 prove**
(Prove-Hadamard ≈ 464 ms vs Prove-NoHadamard ≈ 46 ms after the fused
coefficient-form prover, commits `9505876`+`7a27606`). Its sumcheck cost is

```
   O(2ⁿ · K · D)  GF(2¹²⁸) multiplications
```

(K = 16 relations, D = 32 bits, n = num_vars). Every round after round 1 runs
in the full extension field GF(2¹²⁸), even though the witness bit-slices
`U_b, V_b, W_b` are **GF(2)-valued on the hypercube**. We are doing big-field
arithmetic on base-field data — the regime small-characteristic sumcheck exists
to fix.

## 2. What Binius does (and we don't)

From the Binius blueprint (`/blueprint/backend/ands`, the "Rijndael zerocheck")
and the literature, Binius64 stacks **three** ideas, none of which we use:

1. **Word-level constraints** — it represents 64-bit words, not bits: "64-fold
   smaller constraint systems than bits alone." We split each operand into
   D = 32 bit-slice MLEs (`σ^b` batches them) — the `×D` factor.
2. **Univariate skip** (Gruen [Gru24]; AND-specific variant Hu et al. [Hu+25]):
   fuse the first `k` Boolean sumcheck variables into **one** univariate round.
   The witness is base-field on `{0,1}^k`, so the bulk of that round's work is
   small-field; extension-field muls drop from ~`n` to ~`n/2k` — **up to 128×
   for GF(2)→GF(2¹²⁸)**.
3. **Small tower-field arithmetic** — per-round work in the 8-bit **Rijndael
   field GF(2⁸)** "where possible," not GF(2¹²⁸).

Sources: Dao et al. eprint 2026/587; Bagad-Domb-Thaler eprint 2024/1046
("Sum-Check over Fields of Small Characteristic"); eprint 2025/719 ("Packed
Sumcheck …"); Gruen [Gru24] eprint 2024/108; Binius blueprint; Irreducible
"Slicing Up Binary Towers".

## 3. Univariate skip on our degree-3 zerocheck (mechanism)

Our claim: `Σ_{x∈{0,1}ⁿ} eq(x,r)·C(x) = 0`, with
`C(x) = Σ_k γ^k Σ_b σ^b (U_b(x)·V_b(x) − W_b(x))` (degree 3: eq linear,
`U·V` quadratic).

**Standard** (today): bind one variable/round, n rounds, degree-3 messages;
after round 1's GF(2¹²⁸) challenge everything is big-field.

**Univariate skip of the first `k` vars:** identify `{0,1}^k` with a
`2^k`-element F₂-affine subspace `H ⊂ GF(2¹²⁸)` via a basis `{β₁..β_k}`
(`(b₁..b_k) ↦ Σ bᵢβᵢ`). A multilinear in `k` vars ↔ a **univariate of degree
`< 2^k`** over `H` (the additive-NTT / subspace-evaluation correspondence).
For fixed `x' ∈ {0,1}^{n-k}`, `C(·, x')` becomes a univariate in `t` of degree
`≈ d·2^k = 3·2^k`. The prover sends

```
   p(t) = Σ_{x'∈{0,1}^{n-k}} eq(t,x';r) · C(t, x')      deg p ≈ 3·2^k
```

(its `3·2^k+1` evaluations), the verifier samples **one** `r* ∈ GF(2¹²⁸)`,
binding all `k` vars at once; the remaining `n-k` rounds are standard.

**Why it's cheaper:** evaluating `p` over the domain splits into
- the `2^k` **subspace points** `H` (= the `{0,1}^k` bit-evaluations): products
  of bits ⇒ **GF(2)/GF(2⁸) arithmetic**, cheap;
- the extra `(d-1)·2^k` **off-subspace points**: GF(2¹²⁸).

So the dominant `2^{n-k}`-fold accumulation runs in the small field for the
subspace part; big-field muls are incurred only on the off-subspace points and
the final fold. That is the `n → n/2k`-style reduction.

## 4. Soundness

- The fused round adds soundness error `≤ deg(p)/|F| = 3·2^k / 2¹²⁸` — negligible
  for any sane `k` (e.g. `k=5 ⇒ 96/2¹²⁸`). Remaining rounds: standard.
- Verifier-visible: the proof now carries a degree-`3·2^k` univariate for the
  fused round (≈ `3·2^k` field elements) instead of `k` degree-3 messages — a
  small proof-size increase, and the verifier evaluates one extra univariate
  (cheap; verifier is not the bottleneck).
- **NOT byte-identical** — so the existing `fast_path_matches_generic` /
  `fused_matches_generic_large` gates do **not** apply. A new gate is needed:
  (a) the univariate-skip discharge accepts an honest trace and the closing
  evals match an independent recompute, and (b) it rejects corrupted traces
  (à la `corrupt_w_is_rejected`).

## 5. Estimated win

The discharge is GF(2¹²⁸)-mul-bound (~2·10⁸ muls at nvars=16). Univariate skip
moves the dominant early-round muls to the small field. Theoretical ceiling for
the ext-field-mul-bound part is the tower depth (**128× GF(2)→GF(2¹²⁸)**);
realistically, for degree 3 with the `K·D`-term comb, off-subspace overhead, and
only the first `k` rounds fused, expect **~4–16× on the discharge** — i.e.
464 ms → roughly **30–120 ms**, removing most of the 92 % bottleneck. (The
exact factor needs the Hu et al. AND-skip cost + a prototype; see §7.) Adding
word-level constraints (drop the `×D=32`) compounds further but is a separate
rearchitecture.

## 6. Effort — LARGE, because the prerequisites don't exist

Grep of `poly`/`piop`/`protocol` confirms the codebase has **none** of:
- an **additive NTT / F₂-subspace evaluation** (the existing `pntt/radix8` is a
  *prime-field* `ark_ff::FftField` NTT for the IPRS code — wrong field);
- a **small tower subfield** (only `binary_gf128/gf192/u64/f2_wide`; no GF(2⁸));
- any **univariate-skip** prover/verifier.

(Binary additive FFT exists only as `documentation/binary-add-fft-doc/main.pdf`,
no source.) So this is a **foundational build**, not an incremental opt:

1. Additive-NTT / subspace evaluate-&-interpolate over GF(2¹²⁸) (and/or a GF(2⁸)
   tower) — the core missing primitive.
2. Univariate-skip sumcheck **prover** for the degree-3 comb (subspace evals,
   small-field accumulation for the `{0,1}^k` part, off-subspace lift).
3. Matching **verifier** (univariate check) + transcript/proof format change.
4. Soundness write-up + the new accept/reject gate (§4).
5. Re-integration with Wiring-R: the final point now has `k` vars collapsed to
   one univariate challenge `r*` — the recombination / closing MLE-evals and the
   `eq` handling must accommodate this (integration risk).
6. (Optional, bigger) small tower-field plumbing so the per-round arithmetic is
   GF(2⁸), and/or word-level constraints to drop the `×D`.

Realistically multi-week, high-surface, soundness-sensitive — essentially
porting a core slice of Binius. The technique is well-established and shipped in
Binius, so the *risk is in our integration*, not the math.

## 7. Recommendation — prototype-gate before committing

Do **not** start the full protocol build on faith (cf. the skip=2 lesson: a
plausible technique with a bad/over-optimistic cost model burned a week).
Instead:

- **Phase 0 (contained, ~1–2 days): a throwaway microbenchmark.** Implement only
  the additive-NTT subspace evaluate + interpolate over GF(2¹²⁸) and use it to
  compute ONE fused round of our comb for one relation at nvars=16, with the
  `{0,1}^k` part in GF(2) and the off-subspace part in GF(2¹²⁸). Measure its cost
  vs the equivalent `k` standard rounds. This nails the **realistic factor** and
  de-risks the additive-NTT — the one thing we can't currently estimate.
- **Gate:** proceed to the full build only if Phase 0 shows ≥~4× on the fused
  rounds (and the additive-NTT is tractable). Otherwise keep the fused
  coefficient-form discharge (−24.5 %, shipped) and revisit if/when a GF(2⁸)
  tower lands for other reasons.
- If we proceed, build it as a **new discharge variant** behind the existing
  size-gate style, validated by the new accept/reject + soundness gate, A/B'd at
  nvars=16/20 before replacing the production path.

**Bottom line:** the direction is right — our all-GF(2¹²⁸) discharge is in the
wrong arithmetic regime, and univariate skip is exactly the fix Binius uses. But
it is a foundational, verifier-visible, multi-week build (no prerequisite
machinery exists), so the disciplined next step is the Phase-0 additive-NTT
prototype to convert "up to 128×, realistically 4–16×" into a measured number
before committing.
