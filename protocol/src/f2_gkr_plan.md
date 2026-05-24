# F_2 SHA-256 — GKR-virtual carries & majorities plan

> **2026-05-24 revision 2 (K-combine).** After Phase-2a-pilot landed
> (commit `d3c4636`: ideal swap from `(X^32 − 1)` to `(X^32)` on all 7
> addition constraints, `shiftl1` compensators, bit-31 mask on
> carries), a deeper look at the constraint structure surfaces a
> stronger move than what the first revision proposed.
>
> **The insight.** Every addition constraint C5–C11 has the shape
> ```
>   target − Σ(inputs) − X·m_1 − … − X·m_k − X·c + comp  ∈  (X^32)
> ```
> The `X·_` terms factor over F_2:
> `X·m_1 ⊕ … ⊕ X·m_k ⊕ X·c = X·(m_1 ⊕ … ⊕ m_k ⊕ c)`. Define
> `K := m_1 ⊕ … ⊕ m_k ⊕ c` (a single 32-bit column). Every
> multi-input addition then collapses to the same 2-input shape
> ```
>   target − Σ(inputs) − X·K + comp  ∈  (X^32),
> ```
> and the Phase-2a math pins `K = ShiftR(1)(target ⊕ Σ(inputs) ⊕ comp)`
> — a linear XOR-of-shifted-primaries followed by a single
> `ShiftR(1)`. *Every* carry/majority column becomes Phase-2a-
> virtualisable; the CSA majority columns disappear from the layout,
> not just from commits.
>
> **What this changes.** Phase 2a's scope grows from 7 cols (just the
> Binius carries) to **13 cols** (all 6 CSA majorities + all 7 carries
> combined into 7 K-columns, all virtualised). The CSA tree in
> witness gen disappears — the prover computes K directly from
> `(target, inputs, comp)` via the ShiftR-of-XOR formula. Phase 2b
> shrinks correspondingly: only the 3 AND-based input columns
> (`W_MAJ`, `W_UEF`, `W_UNEG_E_G`) remain as degree-2 work, because
> they're independent witnesses used as *inputs* to additions rather
> than intermediates *of* additions.
>
> Numerical sanity check (4-input add, `x_0=x_1=x_2=x_3=1`):
> chained-add carries `c_1=1, c_2=0, c_3=3`, so
> `K = c_1 ⊕ c_2 ⊕ c_3 = 2`. Direct formula:
> `target=4, Σx=4 (≡0 in XOR), comp=0`, so
> `ShiftR(1)(target ⊕ Σx) = ShiftR(1)(0b100) = 0b010 = 2`. ✓
>
> Soundness story unchanged: as for any single-carry virtualisation,
> the constraint pins `K` given `(target, inputs, comp)` but doesn't
> on its own enforce `target = Σ inputs mod 2^32`. The outer SHA-256
> boundary check (public final-hash vs prover's claimed hash) is
> what makes the prover have to use the honest target everywhere
> upstream.
>
> **Revised phase table** (replaces both prior tables — including
> the one in revision 1 below):
>
> | Phase | What | Mechanism | Cols eliminated | Est. |
> |-------|------|-----------|------------------|------|
> | 0 ✅ | Sklansky reference + golden vectors | n/a | — | done |
> | 1 ✅ | GKR substrate (single 32-bit add, GF(2^192)) | layered sumcheck | — | done |
> | 2a-pilot ✅ | `(X^32)` ideal swap + `shiftl1` comps + bit-31 carry mask | constraint refactor | 0 (sets up 2a-K) | done (commit `d3c4636`) |
> | **2a-K (new)** | **K-combine refactor: replace `(m_1…m_k, c)` per add with one `K` col; declare all 7 K's virtual via Rot/ShiftR-of-XOR-of-shifted-primaries spec** | **linear virtual (no GKR)** | **13** (6 CSA majs + 7 carries) | **2–4 d** |
> | 2b | Degree-2 virtualisation for the 3 input AND/Maj cols (`W_MAJ`, `W_UEF`, `W_UNEG_E_G`) — one Hadamard-style sumcheck per column, reusing the Phase-1 sumcheck plumbing as a one-layer GKR | one-layer GKR per col | 3 | 2 d |
> | 3 | Cross-instance batching across all 2b instances in a SHA block | random linear combination | — | 1 d |
> | 4 | Bench + ship | — | — | ½ d |
>
> **Total eliminable witness cols: 16 of 27** (= 59%), up from the
> revision-1 estimate of 16 split awkwardly between linear (7) and
> degree-2 (9). The mechanism split is now clean: 13 via Phase 2a's
> linear virtualisation alone, 3 via a tiny degree-2 sumcheck.
>
> **Phase 2a-K work breakdown:**
> 1. **Column-layout update** in [`test-uair/src/sha256_f2.rs`](../../test-uair/src/sha256_f2.rs):
>    drop 6 `W_M_*` slots, rename 7 `W_C_*` slots → `W_K_*`. Witness
>    count: 27 → 21.
> 2. **Constraint code update**: collapse each C5–C11 to the unified
>    `target − Σ inputs − X·K + comp ∈ (X^32)` template.
> 3. **Witness-gen update**: replace CSA-tree computation with direct
>    `K = ShiftR(1)(target ⊕ Σ inputs ⊕ comp)`. The intermediate
>    `(s, m_1, m_2, …)` chain disappears entirely — `target` is still
>    `wrapping_add` of the inputs, but no CSA bookkeeping.
> 4. **Compensator update**: the `shiftl1`-of-carry term in
>    `pa_c_*_vals` now uses K directly (one term instead of
>    `Σ shiftl1(m_j) ⊕ shiftl1(c)`).
> 5. **Virtualisation-mechanism extension** in
>    [`protocol/src/f2_prove.rs`](f2_prove.rs): introduce
>    `F2RotShiftedXorVirtualSpec { rot_amount, sources: Vec<(col_idx, row_shift)> }`
>    (or cascading of existing `F2VirtualBpSpec` → `F2BitOpVirtualSpec`,
>    pick whichever the existing PCS spot-check is easier to extend).
>    The verifier reconstructs the virtual cell from sampled-position
>    source cells (possibly at multiple rows when `row_shift ≠ 0`).
> 6. **Declare the 7 K cols virtual** with their derivation specs and
>    confirm all 13 baseline tests + the SHA-256 roundtrip still pass.
> 7. **Bench**: rerun `Steps` + e2e at nvars=9, compare against the
>    `phase2a-pilot` baseline already captured in `target/criterion/`.
>
> **Risks / off-ramps:**
> - **Cross-row sources.** The XOR sources for some K cols include
>   shifted versions (e.g. `W_W^{↓16}`, `W_A^{↓4}`). The existing
>   shift infrastructure declares these via `ShiftSpec` and they're
>   accessible inside `constrain_general` — but the virtualisation
>   mechanism today (`F2BitOpVirtualSpec` / `F2VirtualBpSpec`) only
>   touches same-row sources. PCS spot-check reconstruction needs to
>   open source cells at multiple rows. If extending the spot-check
>   for cross-row sources turns out fiddly, fall back to (a) only
>   virtualising the K's whose sources are all same-row (C7, T_2), and
>   (b) keeping cross-row-source K's committed until a follow-up.
> - **Boundary rows.** The K column on inactive rows is whatever the
>   `ShiftR(1)(target ⊕ Σ inputs ⊕ comp)` formula yields — the comp
>   column already absorbs the residue there. Confirm in tests that
>   inactive-row K values don't break the proof's selector-gated
>   constraints (C16–C18).
> - **Sumcheck cost.** Virtualised cols still participate in
>   sumcheck (they have MLE-eval claims). Eliminating 6 CSA cols
>   reduces the γ-batched col count from 41 → 35, which actually
>   *speeds up* sumcheck. Eliminating 7 K cols from commits speeds
>   up commit + PCS open.
>
> ---
>
> **2026-05-24 revision 1 (superseded by revision 2 above).** After
> building Phase-0/1 (Sklansky reference + a standalone GKR substrate
> over GF(2^192)) I went deeper into the F_2 prover and found two
> things that substantially change the downstream phases:
>
> 1. **The carry columns are linear in bits.** C10/C11 (and analogously
>    C5–C9) force the carry word to be a **Rot(31) of an XOR-of-other-
>    columns**, e.g.
>    `c_FF_A = Rot(31)(W_A^{↓4} ⊕ W_A ⊕ PA_A ⊕ PA_C_FF_A)`. Proof: for
>    a degree-32 polynomial to lie in `(X^32 − 1)` its bit-32 must equal
>    its bit-0, and bits 1..31 must vanish — solving those constraints
>    explicitly yields the cyclic-shift identity. The 7 Binius carries
>    are all of this shape (the chained-add ones reference the CSA
>    majority columns as sources, which stay committed in the linear-
>    virtualisation pass).
> 2. **The F_2 prover already has a virtualisation mechanism**
>    ([`F2VirtualBpSpec`](src/f2_prove.rs#L176) for XOR-of-primaries,
>    [`F2BitOpVirtualSpec`](src/f2_prove.rs#L219) for Rot/ShiftR of a
>    single primary). Both skip the Merkle commitment and PCS open
>    cost — the virtual column gets its own MLE-eval claim during
>    sumcheck (so the protocol path through IC + sumcheck is unchanged)
>    but never enters the codeword/Merkle pipeline. Encoding
>    consistency at PCS spot-check time re-derives the virtual cell
>    from its sources at the bit level.
>
> Together these two facts mean **the 7 carries can be eliminated with
> no GKR at all** — they just need the existing virtualisation
> mechanism extended to support "Rot of XOR-of-shifted-primaries"
> (currently each spec type does just one piece). That's the biggest,
> simplest win and should run first.
>
> The Phase-1 GKR substrate isn't wasted, but its actual customer is
> the **non-linear** column groups: CSA majorities (6 cols) and
> AND-based Maj/Ch (3 cols), all degree-2 in bits. The Sklansky
> machinery is overkill for those — they want a single degree-2
> sumcheck per column, not a 5-stage layered protocol. So the GKR
> ends up looking more like "one-layer GKR (= a Hadamard-style sumcheck)
> per column", and the depth-5 Sklansky construction stays in the
> codebase as documented dead-code unless and until we tackle a more
> complex non-linear circuit.
>
> Revised phase plan (replaces §3 below from Phase 2 onward; §0–1 are
> done as written):
>
> | Phase | What | GKR? | Eliminates | Est. |
> |-------|------|------|------------|------|
> | 0 ✅ | Sklansky reference + golden vectors | n/a | — | done |
> | 1 ✅ | GKR substrate (single 32-bit add) | yes | — | done |
> | 2a (new) | Linear-virtualisation extension: Rot-of-XOR-of-shifted-primaries virtual spec; declare it for all 7 Binius carries in `sha256_f2.rs` | **no** | 7 carry cols | 1–2 d |
> | 2b (new) | Degree-2 virtualisation: one-layer GKR (= Hadamard) for `Maj(a,b,c) = ab ⊕ ac ⊕ bc` and `AND(a,b) = ab`; declare it for `W_MAJ`, `W_UEF`, `W_UNEG_E_G` and the 6 CSA majorities | yes (1-layer) | 9 Maj/AND cols | 2 d |
> | 3 (was 4) | Decide whether the deferred-Hadamard plumbing already exists or needs implementing for 2b | — | — | ½ d |
> | 4 (was 6) | Cross-instance batching across all 2b instances in a SHA block | yes | — | 1 d |
> | 5 (was 7) | Bench + ship | — | — | ½ d |
>
> The original Phase 2–6 below is preserved as the *as-planned* record;
> read it together with this revision header to see what's changed and
> why. The Sklansky-tree GKR remains a building block we may want to
> revive if a future workload introduces a non-linear circuit deeper
> than degree 2 (e.g. addition modulo a non-power-of-two).

---

## Original (pre-revision) plan


Goal: eliminate the **13 committed witness columns** in
[`test-uair/src/sha256_f2.rs`](../../test-uair/src/sha256_f2.rs) that
exist only to evidence carry / majority arithmetic, replacing them with
"virtual" wires proven on the fly by a small, batched GKR-style
sumcheck over an F_2 arithmetic circuit. No new column commitments,
trade ~32·n committed bits per gate for ~5 extra sumcheck rounds per
gate, batched across all gates in one transcript pass.

This is a **test/POC plan**: each phase is small, instrumented against
a baseline, and has a clear off-ramp if the numbers don't move.

---

## 0. Targets

Columns to remove from
[`sha256_f2.rs:171-185`](../../test-uair/src/sha256_f2.rs#L171):

| Group              | Cols                                       | Count |
|--------------------|--------------------------------------------|-------|
| Binius carries     | `W_C_W/T1/T2/A/E/FF_A/FF_E`                | 7     |
| CSA majorities     | `W_M_W1/W2`, `W_M_T1_{1..4}`               | 6     |
| Maj / Ch operands  | `W_UEF`, `W_UNEG_E_G`, `W_MAJ`             | 3     |

That's 16 of 27 witness columns. Phase order below keeps the
"deferred external Hadamard" group (the last 3) for last, because
those interact with how the prover currently leaves Ch/Maj
unconstrained.

---

## 1. Math reminders

**Parallel-prefix carry over F_2.** For a 32-bit add `c = a + b mod 2^32`:
per-bit `G_i = a_i·b_i`, `P_i = a_i ⊕ b_i`; compose blocks
associatively with
`(G,P) ∘ (G',P') = (G' ⊕ P'·G, P'·P)`
(`⊕` coincides with OR here since `G'=1 ⇒ P'=0`). Fold 32 bits with a
Sklansky tree — **depth 5**, degree-2 gates. Output stream `k_i` feeds
`c_i = a_i ⊕ b_i ⊕ k_i`.

**Maj / CSA majority.** `M(x,y,z) = xy ⊕ xz ⊕ yz` — **depth 1**,
degree-2.

**Carry-save adder layer.** For a 3-input add: `(s, m) = (x⊕y⊕z,
M(x,y,z))`, then the "next layer" treats `s + 2·m` as the new
2-input sum. For the existing 4-input message schedule and 6-input
T_1, the current code already factors as a CSA tree of depth 2 and 4
respectively, ending in one Binius adder — that structure stays; we
just stop committing the per-layer majorities and final carry.

---

## 2. Architectural integration

The existing modular-add constraints
([`sha256_f2.rs:388-466`](../../test-uair/src/sha256_f2.rs#L388))
have shape
```
target − Σ(inputs) − Σ X·m_j − X·c + compensator  ∈  (X^32 − 1).
```
After ideal-check + ψ_α projection, sumcheck sees claims on the MLEs
of every term, including `m_j` and `c`. Removing the commitments
means: at the point sumcheck would consume those claims, we **derive
them** from claims on the input MLEs via a GKR pass over the carry/Maj
circuit.

Concretely, the integration point is one new claim-producing layer
between [`protocol/src/f2_native_ic.rs`](src/f2_native_ic.rs) (which
already produces post-ψ_α claims per column) and the multi-degree
sumcheck in [`piop/src/sumcheck/multi_degree.rs`](../../piop/src/sumcheck/multi_degree.rs).
For each virtualised column, the verifier needs the MLE evaluation of
that column at the sumcheck random point `r`. GKR proves that value
equals a known circuit evaluated on the (still committed) input MLEs
also at `r`.

Two consequences:

- **No change to the IC stage.** The constraint polynomial keeps its
  current shape; the carry and majority terms keep their slots. We
  simply mark those slots as "virtual" so the prover doesn't commit
  them and the verifier reroutes its claim.
- **GKR's bottom-layer opening is a claim on the committed input
  MLEs at `r`.** This rides on the same MLE-opening lane being built
  in [`f2_open_plan.md`](src/f2_open_plan.md) — no second lift-and-
  project mechanism is needed.

---

## 3. Phase plan

### Phase 0 — Spec & golden vectors (≈ ½ day)

- Document the Sklansky 32-bit prefix layers (which pair-indices
  compose at each depth) as constants alongside the existing
  `binius_carry` helper at [`sha256_f2.rs:595`](../../test-uair/src/sha256_f2.rs#L595).
- Add a `#[test]` fuzz vs. `u32::wrapping_add` for the layered
  carry function and Maj.
- **Exit criterion:** byte-exact match with the existing
  `binius_carry` for 10⁶ random `(a,b)`.
- **Files:** new `test-uair/src/sha256_f2_prefix.rs`.

### Phase 1 — GKR substrate (≈ 2 days, biggest risk)

There is no layered-sumcheck code in the repo today (Explore
confirmed only flat multi-degree sumcheck and the IC fold exist).
Build the minimum:

- A `GkrLayer` trait with a per-layer sumcheck of degree 2 (Maj/carry
  layers never exceed that) and the standard "two-into-one" trick
  for the next-layer claim (one sumcheck eval becomes the next
  layer's claim via a Fiat-Shamir random combiner).
- A `GkrProver` / `GkrVerifier` pair that consumes a slice of
  `GkrLayer`s, threads claims, and emits a final claim on the bottom
  inputs.
- Build on `piop::sumcheck::multi_degree` so we get the GF(2^192)
  transcript glue for free.
- **Exit criterion:** end-to-end test on a single 32-bit add — prove
  a claim on `c`'s MLE at random `r` from claims on `a, b`'s MLEs at
  `r`, with the 5-layer Sklansky tree. Compare against an explicit
  MLE evaluation of `c`.
- **Files:** new `piop/src/gkr/{mod,layer,prover,verifier}.rs`.

### Phase 2 — Eliminate the simplest carry (`W_C_FF_A`, `W_C_FF_E`) (≈ 1 day)

The feed-forward carries (C10, C11 at
[`sha256_f2.rs:451,460`](../../test-uair/src/sha256_f2.rs#L451)) are
the cleanest 2-input adds in the system — no CSA layer above them.
Use them as the minimal end-to-end demo.

- Add a `virtual_columns` set to the UAIR signature; gate witness
  generation and commitment on membership.
- Wire the IC stage to skip the column's commitment, then dispatch
  GKR for its claim after IC + ψ_α.
- Adapt the bench
  [`protocol/benches/f2_sha256.rs`](benches/f2_sha256.rs) to report
  prover time, proof size, and verifier time **broken down between
  commit / IC / sumcheck / GKR** so we can read off the trade.
- **Exit criterion:** verifier accepts; prover loses ~2 column
  commits (~64·n bits) and gains ~10 sumcheck rounds; net wall-time
  is at worst neutral on the 7-compression bench.

### Phase 3 — Eliminate Maj / Ch operands (`W_UEF`, `W_UNEG_E_G`, `W_MAJ`) (≈ ½ day)

These are depth-1 degree-2 gates with the current "deferred external
Hadamard" placeholder. With Phase 1 in hand, this is a one-layer GKR
pass per gate.

- Drop their commitments; replace the deferred external check with
  a degree-2 sumcheck against the already-committed inputs `(e,f)`,
  `(¬e,g)`, `(a,b,c)`.
- **Bonus:** this finally closes the deferred-Hadamard gap flagged at
  [`sha256_f2.rs:60-64`](../../test-uair/src/sha256_f2.rs#L60).
- **Exit criterion:** identical to Phase 2; also remove the
  "deferred" note from the doc-comment.

### Phase 4 — Eliminate Binius carries on chained adds (`W_C_W/T1/T2/A/E`) (≈ 1–2 days)

Five more carries, all on adds whose inputs (the residuals of CSA
trees) are not directly committed — they live as intermediate wires
of the CSA stage. Two options:

- **(a) Re-prove from primitive inputs.** Recompute the whole tree
  (CSA layers + final prefix-add) as one tall GKR. Depth = log₂(k) + 5
  where k is the fan-in (4 for `W_C_W`, 6 for `W_C_T1`, 2 for the
  rest). All gates degree 2.
- **(b) Keep the CSA `m_j` claims as virtual but feed them into the
  Binius-carry GKR via separate one-layer GKRs.** Smaller transcripts
  per gate but more bookkeeping.

Recommend **(a)**: simpler, batches naturally with Phase 5, and the
total depth is still ≤ 7 layers for the worst case (T_1).

- **Exit criterion:** all 7 carries gone; per-row commit goes from 41
  → 34 (and Phases 5/6 push it lower).

### Phase 5 — Eliminate CSA majorities (`W_M_W1/W2`, `W_M_T1_{1..4}`) (≈ ½ day)

If Phase 4 picked option (a) above, these come for free — the CSA
layers are already part of the layered circuit and their majorities
are wire values, not commitments. If (b), do a one-layer GKR per
majority as in Phase 3.

### Phase 6 — Cross-add batching (≈ 1 day)

Naive Phase 2–5 runs one GKR per (add × row). For a 7-compression
bench that's ~7·64·11 = ~5000 small GKRs. Batch:

- Take a random linear combination over all (gate, row) instances at
  each circuit depth. Result: **one sumcheck per circuit depth**, not
  one per gate. Total transcript: ~7 sumcheck rounds for the deepest
  layer instead of ~5000·7.
- The bottom-layer claim becomes a single multilinear evaluation on
  each input column at `r`, which is exactly what the MLE-opening
  lane already wants.
- **Exit criterion:** prover wall-time strictly below the
  pre-Phase-2 baseline at `num_vars = 9` (the bench shape at
  [`f2_sha256.rs:26-30`](benches/f2_sha256.rs#L26)).

### Phase 7 — Bench, decide, write up (≈ ½ day)

- Run the existing `f2_sha256` bench with three configs: baseline,
  Phase-3-only (Maj eliminated), full (Phases 2–6).
- Report: commit time, IC time, sumcheck time, GKR time, total
  prover, proof size, verifier time.
- **Decision criterion:** ship if full config beats baseline on
  prover wall-time **and** proof size at `num_vars ≥ 9`. Otherwise
  ship only the wins (almost certainly Phase 3 at minimum, since
  it costs ~3 commitments and one degree-2 sumcheck).

---

## 4. Risks / off-ramps

- **GKR substrate is from-scratch (Phase 1).** Mitigation: keep the
  trait surface small, lean on the existing multi-degree sumcheck.
  Off-ramp: if Phase 1 blows past 3 days, ship only Phase 3 with a
  hand-rolled one-layer sumcheck — no general GKR needed for that.
- **MLE-opening lane is in flight
  ([`f2_open_plan.md`](src/f2_open_plan.md)).** GKR's bottom claim
  must land in the same shape that lane consumes. Mitigation: Phase 1
  hits a stub claim sink; only Phase 2 onward wires to the real
  opening lane, by which time `f2_open_plan` should be settled.
- **Batching in Phase 6 may degrade soundness if we get the random
  combiner wrong.** Mitigation: follow the standard "λ-power
  combiner" used in vanilla GKR, with the λ drawn from the same
  GF(2^192) transcript already in use.
- **Verifier cost.** GKR adds log-depth × constant rounds per
  virtualised column. For a verifier already paying for the
  multi-degree sumcheck, this should be a rounding error; confirm in
  Phase 7.

---

## 5. Work-breakdown summary

| Phase | New files                                   | Modified files                                       | Est. |
|-------|---------------------------------------------|------------------------------------------------------|------|
| 0     | `test-uair/src/sha256_f2_prefix.rs`         | —                                                    | ½ d  |
| 1     | `piop/src/gkr/{mod,layer,prover,verifier}.rs` | `piop/src/lib.rs`                                  | 2 d  |
| 2     | —                                           | `test-uair/src/sha256_f2.rs`, `protocol/src/f2_native_ic.rs`, `protocol/src/f2_prove.rs`, `protocol/benches/f2_sha256.rs` | 1 d  |
| 3     | —                                           | same as Phase 2                                      | ½ d  |
| 4     | —                                           | same as Phase 2                                      | 1–2 d|
| 5     | —                                           | same as Phase 2 (mostly deletions)                   | ½ d  |
| 6     | —                                           | `piop/src/gkr/prover.rs`, `verifier.rs`              | 1 d  |
| 7     | —                                           | bench only                                           | ½ d  |

Total: **6–7 working days** with clean off-ramps after Phases 1 and 3.
