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

## 8. RESOLUTION (got the Dao paper via `pdftotext`) — the technique is BYTE-IDENTICAL, and we built it in the WRONG FIELD

The §1–7 framing above over-indexed on Binius's *verifier-visible* univariate
skip. The Dao et al. paper (eprint 2026/587, extracted) shows the relevant
technique is its **byte-identical cousin, the small-value sum-check prover**, and
gives the exact cost:

- **Speedup `Θ((d²κ)^{1/δ})`, `δ = log₂(d+1)`** (paper §4–5, l.1156). For our
  **d=3 ⇒ δ=2 ⇒ speedup = 3√κ.**
- **`κ ≈ N^{log₂3}` for tower fields with Karatsuba** (l.1152), `N` = extension
  degree of big-field over small-field. So:
  - small = GF(2): `N=128 ⇒ κ≈2180 ⇒ ~140×`, optimal skip `v*=log₄(9κ)≈7`;
  - small = GF(2⁸): `N=16 ⇒ κ≈81 ⇒ ~27×`, `v*≈5`.
- The off-hypercube extrapolation is **`Θ(d²)` `sb` (small×big) muls** (l.744),
  *not* bb — this is the sign error in my §8-draft model: the witness stays
  small, so a bit×GF(2¹²⁸) "mul" is a **select/conditional-XOR**, ~free, not a
  CLMUL. Plus **delayed reduction** in the extension (l.609–620): accumulate
  unreduced degree-`(2κ−2)` products, reduce once.
- **It is verifier-preserving** (the paper's whole point vs the univariate skip):
  the proof/degree are unchanged — so the existing byte-identity gate applies.

**Why our removed skip prover (ce177a3..f615f7f) was 40–74× slower:** we built
exactly Procedure 1 + the v-variate prefix, but executed **every** operation in
GF(2¹²⁸). The 3√κ speedup is in **bb-equivalents** — it only appears if the
early-round `{0,1}^v`-grid arithmetic is done in the **small field** (bit-packed
GF(2) AND/XOR for ss, select for sb). We did ss in bb ⇒ all of the algorithm's
*larger* op-count, none of the √κ discount ⇒ slower. Also, Procedure 1's
*degree* win (Θ(d²)→Θ(d log d)) only helps `d≥4` (l.752); our `d=3` gets ~nothing
from that sub-part — but the **small-value (ss/sb) win is separate and applies at
d=3** (the 3√κ above).

**Revised effort — MUCH smaller than §6 said, and byte-identical:**
- No additive NTT, no GF(2⁸) tower *required* (GF(2) bit-packing suffices for
  ss; GF(2⁸)/GFNI is an optional further lever), no verifier change, no new
  soundness argument — the existing `fast_path_skip2_matches_generic`-style
  byte-identity gate covers it.
- The algorithm skeleton is recoverable from git (`ce177a3..f615f7f`).
- The real work: reimplement the prefix's early-round arithmetic in **bit-packed
  GF(2)** (ss = packed AND/XOR; sb = select into a GF(2¹²⁸) accumulator) with
  **delayed reduction**, and pick `v` (~5–7). This is the thing the removed
  version never did.

**Caveats (why measure, not trust the 140×):** the speedup is asymptotic
bb-equivalent. On aarch64 we have PMULL for GF(2¹²⁸) but **no GFNI/native tower
mul** (paper l.2498–2505), so realizing cheap ss/sb depends on bit-packing
quality; the off-hypercube grid is `Θ((d+1)^v)=Θ(d²κ)` cells (space ~20K at
v=7) and its sb/bb tail is real; raw op-count is large even if bb-equivalent is
small. Realistic win is "large but < 140×" — exactly what Phase-0 must measure.

## 9. Phase 0 (REVISED) — measure the bit-packed small-value prefix

Drop the additive-NTT/univariate-skip prototype (wrong technique). Instead:
- **Resurrect** the removed prefix (`git show ce177a3..f615f7f`) as a throwaway,
  and reimplement *just* its hottest path — the `{0,1}^v`-grid accumulation — in
  **bit-packed GF(2)** (ss = u64/SIMD AND+XOR; sb into a GF(2¹²⁸) accumulator via
  select), with delayed reduction, for `v≈5` at nvars=16, one relation.
- **Measure** it (bb-equivalent *and* wall-clock) vs the `v` standard GF(2¹²⁸)
  rounds it replaces. This directly tests whether the ss/sb arithmetic is cheap
  enough on aarch64 to realize a meaningful fraction of 3√κ.
- **Gate:** proceed to the full byte-identical small-value discharge only if the
  prefix shows ≥~4× on the skipped rounds. (It's byte-identical, so integration
  reuses the existing discharge verifier + recombination — far less than the
  univariate-skip build in §6.)

## 10. Phase 0 — ✅ DONE: the inner-comb ceiling is ~19× (clears the gate)

Probe shipped as `poly/benches/binary_gf_compare.rs::bench_discharge_comb`
(`binary_gf/discharge_comb_512term`), the discharge comb at one sumcheck point
(K=16 relations × D=32 bits = 512 terms), two ways:

| arm | what | time/comb |
|-----|------|-----------|
| `bb_GF128_products` | standard: post-fold slices are general GF(2¹²⁸) ⇒ 512 `U_b·V_b` are full CLMUL muls (~1041 bb) | **2594 ns** |
| `sv_packed_psi_x4`  | small-value: operands are bits ⇒ `(U_k&V_k)⊕W_k` packed AND/XOR + `Σ_b σ^b(·)` = ψ_σ via the x4 NEON kernel + 17 bb | **137 ns** |

**⇒ ~19× faster** (M-series, `target-cpu=native`, `--features simd`). Breakdown:
bb ≈ 2.5 ns/GF128-mul (CLMUL-throughput-bound); sv's 137 ns ≈ 17 bb-muls (~42 ns)
+ 4 ψ-x4 calls (~95 ns ⇒ ψ_σ(32-bit) ≈ 6 ns ≈ 2.4 bb-equiv, vs the ~64 bb it
replaces per relation). This is the realized `√κ` on the per-point inner comb,
and it **clears the ≥4× gate with margin** — confirming the removed skip prover's
failure was the `bb` execution, not the algorithm.

**Caveat — this is the CEILING.** It measures only the inner comb (the part the
small-value prover does cheaply). The full prover adds: (a) the off-hypercube
grid `Θ((d+1)^v)` cells (sb/bb extrapolation via Procedure 1 — `O(d log d)` bb,
small for d=3 but nonzero), (b) prefix construction + delayed-reduction
bookkeeping, (c) only `v≈5–7` of the `n` rounds are skipped (the tail stays
standard, but it's `2^v`-cheaper). So the **net** discharge speedup will be
< 19× — plausibly **~5–15×** if the grid overhead stays sub-dominant (Procedure
1), i.e. discharge 418 ms → ~30–80 ms, nvars=16 prove → ~75–125 ms (a ~4–6× total
prove win), byte-identical. Phase 1 must measure the *full* prefix (grid
included) to confirm the net.

**Recommendation (Phase 0): cleared the *ceiling* gate — but see §11, which
walks it back after modelling the off-hypercube grid.**

## 11. NET analysis (Phase-1 entry) — the ceiling is misleading for d=3; the byte-identical path is ~neutral

Starting Phase 1, the first task is the *full* v-variate prefix, and modelling
its dominant cost — the **off-hypercube grid** — reverses the Phase-0 optimism:

- The v=2 prefix is a grid of `(d+1)² = 16` cells; only `2² = 4` are boolean.
  The other **12 are off-hypercube**, where the operands are the *extrapolated*
  multilinear values (not bits), so `U_b·V_b` there is a genuine field product,
  not a free AND. Phase-0's 19× applies only to the 4 boolean cells.
- Procedure 1's cost is `O(d^v)` **bb** per multiproduct (paper l.804). For our
  comb that's per `(relation,bit)`: ~`3²=9` bb for `eq·U·V` + ~`2²=4` bb for
  `eq·W` ≈ **13 bb/term**, vs the standard round's **~6 bb/term** (coeff-form).
  Only the boolean base is `ss`. So at **d=3 the off-hypercube `bb` work
  dominates** and the `ss` saving is a minority ⇒ net **~neutral-to-worse**, not
  19×. The `3√κ` asymptotic needs the boolean cells to be a *majority* of the
  Procedure-1 work, which happens at **higher d**, not d=3.
- This reconciles everything: (a) the removed GF(2¹²⁸) prefix measured
  "~neutral"; (b) the Dao headline wins (10.9×) are **high-degree** Spartan; (c)
  Binius's AND win is **word-level (no `×D=32` bit-split) + the verifier-visible
  univariate skip + GF(2⁸)**, *not* the byte-identical small-value prover.

**Corrected conclusion:** the **byte-identical** small-value prover is **not the
lever for our d=3 discharge** — it is ~neutral. The real Binius-scale win is the
**verifier-visible rearchitecture**: (1) **word-level** AND constraints to drop
the `×D=32` bit-slice factor (Binius's "64-fold smaller than bits"), and (2) the
**univariate skip** — i.e. §1–6 of this doc, the *big, foundational,
verifier-visible* build (new sumcheck protocol + verifier + soundness + GF(2⁸)
tower + word-level constraint reshape). That is a multi-week, research-grade
effort, not an incremental optimization.

**Cheapest way to confirm before committing to the big build:** resurrect the
removed GF(2¹²⁸) v=2 prefix (`git show f615f7f`) and measure its *actual* net at
nvars=16 vs the shipped fused evaluator. If "~neutral" as modelled, the
byte-identical path is closed and only the §1–6 rearchitecture remains. (Caveat:
my cost models have oscillated this session; the `O(d^v)`-bb argument above is
the most grounded, and it agrees with the prior empirical "~neutral".)

**Net for this session:** shipped −24.5% byte-identical discharge
(`9505876`+`7a27606`); validated that the small-field *inner comb* is 19×
cheaper (Phase 0); and established that converting that into a *net* discharge
win at d=3 requires the verifier-visible word-level + univariate-skip
rearchitecture — the byte-identical shortcut does not pay. Treat the full build
as a scoped, dedicated future effort guided by Binius's AND reduction + Dao
Procedure 1.

--- (historical: original §8 blocked-state notes follow) ---

### Original §8 (now superseded by §8 RESOLUTION above)

Attempting to scope the Phase-0 prototype surfaced a **mechanism gap**, not just
a missing primitive. A first-principles cost reconstruction of the fused round
for our degree-3, `K·D = 512`-term comb gives the **wrong sign**:

- Fused round (fuse first `k` vars): the `2^k` *subspace* points are cheap (bit
  products on `{0,1}^k`), but the `(d−1)·2^k` *off-subspace* points still seem to
  need, per `(x', relation, bit)`, (i) an additive-NTT extrapolation of each
  slice's degree-`<2^k` univariate `O(d·2^k·log)` and (ii) a GF(2¹²⁸) product —
  totalling `≈ (d·log + d−1)·2ⁿ·K·D` big-field muls.
- Standard first-`k` rounds: `≈ Σ_{i=2}^k 2^{n−i}·K·D·6 ≈ 3·2ⁿ·K·D` big-field
  muls (round 2 dominates).
- ⇒ my model says the fused round is **~8× WORSE**, contradicting the
  established "univariate skip ⇒ up to 128× *fewer* extension-field muls."

So I am **missing the real mechanism** — most likely how the off-subspace /
product work stays small-field. The prime suspect is **packing the base-field
witness into the tower** (Bagad et al. "Packed Sumcheck", eprint 2025/719; Hu et
al. AND variant), so a single GF(2¹²⁸) mul does `2k` base-field lanes at once and
the off-subspace products batch — that is what would flip the sign and yield the
tower-depth (≤128×) win. I could not confirm this: the authoritative PDFs (Dao
2026/587, Bagad 2024/1046, Gruen 2024/108, Hu 2025) do **not text-extract** via
the available web tool (binary streams / HTTP 403), and the HTML blogs don't
cover the product cost.

**A prototype built from a misunderstood mechanism would give a misleading
benchmark.** Before any Phase-0 code, resolve the product/packing mechanism —
options: (a) obtain readable text of Bagad 2024/1046 §(packed) + Hu 2025 / Dao
2026/587 §5; (b) a domain expert states how the off-subspace product cost is kept
small-field. **Then** the contained Phase-0 (additive-NTT subspace
evaluate/interpolate + one fused round, measured vs `k` standard rounds) becomes
trustworthy. The additive NTT itself is well-defined and correctly implementable
regardless; benchmarking it standalone gives the primitive's cost floor but NOT
the end-to-end speedup (which needs the packing mechanism).
