# Lookup methods for the Zinc+ integer path

Branch: `main-beta-lookup` (off `main-beta`). Status: **M2 (LogUp-GKR) IMPLEMENTED
+ e2e-validated** (2026-07-18) — the general primitive is now wired into the
protocol crate (gkr_logup module + bin_multipoint_reducer + step4b/step7 seam)
and passes BinLookup16 e2e (G=1 fast path + G≥2 reducer) plus a tamper-reject
test, with the full protocol suite 23/23 green. The R1 range-check (see
`range-check-design.md`), Regime-A systematization, and lookups on the real
SHA-256 UAIR remain future work. Original design menu below.

This note lays out a *menu* of ways to do lookups / range checks on the
integer path (`Z[X] → φ_q → F_q[X] → ψ_α → F_q`), the setting that filters
that menu, the prior art already in this repo, and a tiered recommendation.

---

## TL;DR

1. **The integer path uses *no* lookup argument today.** Every table-like
   obligation in the SHA-256 UAIR is discharged *algebraically* — through the
   existing CPR sumcheck (bit-op virtual columns) and the booleanity sumcheck
   (bit-polys + "Table-9" virtual booleanity residuals). `signature()` returns
   an empty `lookup_specs` (`test-uair/src/sha256.rs:500`). This is cheap
   (commits nothing extra) but **bespoke per-UAIR** and leaves documented
   soundness caveats.
2. **So "how to do lookups" splits into two regimes.**
   *Regime A* (tiny, bit-structured tables: the SHA μ-carries, Ch/Maj) is
   already covered by algebra; a real lookup argument is optional and probably
   not worth it. *Regime B* (large / arbitrary / non-bit-structured tables:
   AES-style S-boxes, ECDSA mod-`n` reductions, genuine large-value range
   checks, future UAIRs) is where the algebraic dodge runs out and a lookup
   argument earns its keep.
3. **The cost model picks the winners.** Transparent hash-based Zip+
   (Brakedown/Ligero) makes *committing extra columns* the expensive resource
   and rules out KZG-based lookups entirely. Over the large prime field `F_q`,
   additive LogUp is sound (unlike the char-2 `f2-clean` path). Net: favor
   **GKR-discharged, no-extra-commitment** methods.
4. **Recommendation.** Regime A → keep/systematize the algebraic discharge
   (M0), optionally as a declarable `∏(v−j)` range primitive that closes the
   μ-carry NYI caveat at ~zero cost. Regime B → adopt the **already-built
   GKR-LogUp** (M2) as the default general primitive; layer **Lasso-style
   decomposition** (M3) for huge structured tables. If one design must serve
   *both* the integer and `F_2` paths, use the **multiplicative grand product**
   (M4) — the only characteristic-agnostic option.

---

## 0. Where we are: the integer path has replaced lookups with algebra

The SHA-256 integer UAIR (`test-uair/src/sha256.rs`) deliberately avoids a
lookup argument. Each obligation that *would* be a lookup in a typical
field-SNARK is instead encoded so it falls out of machinery the prover already
runs:

| Obligation | How it's discharged today | Where |
|---|---|---|
| Rotations / shifts (`Σ₀,Σ₁,σ₀,σ₁`) | **CPR bit-op virtual columns** (`BitOp::Rot`, `BitOp::ShiftR`) — no committed operands | `sha256.rs:511-514` |
| `F_2[X]`→`Q[X]` mod-2 reduction | per-coefficient **overflow bit-poly witnesses** `ov_*`, pinned by booleanity | `sha256.rs:77-101` |
| `Ch`, `Maj` (bitwise AND) | **Table-9 virtual booleanity residuals** `r_ch1,r_ch2,r_maj` — virtual MLEs pinned by the *same* booleanity sumcheck, plus public edge-row compensators | `sha256.rs:122-157, 558-648` |
| μ-carry ranges (`μ_W∈[0,3]`, `μ_a∈[0,6]`, `μ_e∈[0,5]`) | **self-enforcing**: booleanity bounds each bit; out-of-range carries force a non-32-bit register caught downstream by `w_a/w_e` booleanity | `sha256.rs:115-120, 322-328` |

The unifying philosophy: **route everything through the CPR and booleanity
sumchecks, commit nothing extra.** That is exactly why the prover stays cheap.

**But the cracks are real, and they motivate a uniform lookup primitive:**

- *Documented soundness caveat* on the mod-2 ideal lift: the per-row parity
  check is not realizable because the random `F`-linear row combination
  destroys the small-integer structure before `IdealCheck::contains`
  (`sha256.rs:103-113`, "out of scope").
- The μ-range argument is a *downstream-catch* argument, not a direct check
  (`:322-328`) — correct as analyzed, but fragile and UAIR-specific.
- The Ch/Maj residual machinery needs **public compensator columns**
  (`PA_R_CH2_COMP`, `PA_R_MAJ_COMP`) for the trace-boundary rows
  (`:1013-1050`) — and a now-**stale** module comment still says Ch/Maj are
  "left as free witness columns (unenforced)" (`:54-58`), which is exactly the
  kind of drift that bespoke per-constraint algebra invites.

A general, declarable lookup/range primitive would (a) cover tables that
*aren't* bit-decomposable, and (b) replace these clever-but-brittle tricks with
one audited mechanism.

---

## 1. The setting, and why it filters the menu

Three properties of the integer path decide which lookup techniques are even
applicable:

1. **Lookups run over a large prime field `F_q`** (after `φ_q` then `ψ_α`).
   ⇒ **Logarithmic-derivative (LogUp) is sound here.** (Contrast: the sibling
   `f2-clean` path is `F_2[X]` / char 2, where additive LogUp is *unsound* —
   even multiplicities cancel; see `documentation/f2x-sha-todo.md` ~L2572.
   Only the *multiplicative* grand product survives there.)
2. **The inner engine is a multi-degree sumcheck** at step 4. ⇒ Any lookup
   must present as one or more degree-`d` zerocheck/sumcheck groups that share
   the step-4 evaluation point `r*`, alongside CPR and booleanity.
3. **The commitment is transparent, hash-based Zip+** (Brakedown/Ligero over
   IPRS/RAA codes), and commit/open is already ~13 ms of the ~40 ms prover.
   Two consequences:
   - **KZG-based lookups are out** (cq, Caulk/Caulk+, Baloo): they need a
     pairing-friendly group and a (universal) trusted setup; adopting them
     forfeits transparency. Ruled out below, not in the menu.
   - **Committing extra columns is the expensive resource.** In Brakedown each
     extra committed column costs one encode *and* one extra revealed leaf at
     each of the ~k spot-check positions (bigger proof, more verifier hashing).
     ⇒ Prefer methods that **discharge via GKR/sumcheck and commit nothing
     extra** (or only the unavoidable multiplicities) over methods that commit
     helper/multiplicity columns.

This is the single most important filter, and it is precisely why the existing
GKR-LogUp chose a "chunks-in-clear, nothing extra committed" design.

---

## 2. What actually needs a lookup — two regimes

**Regime A — tiny, bit-structured tables.** The SHA μ-carries (ranges ≤ 6) and
Ch/Maj (bitwise AND). Already handled by §0's algebra. The tables are so small,
or so directly bit-decomposable, that a multiset/permutation lookup argument is
strictly more machinery than the obligation deserves.

**Regime B — large / arbitrary / non-bit-structured tables.** Where the algebra
runs out and a genuine lookup argument pays for itself:
- **AES-style S-boxes** and other non-affine byte maps (no clean bit-residual).
- **ECDSA mod-`n` reductions** and big-field byte/range tables (the part of
  ECDSA verification beyond the MSM that's still future work).
- **Genuine large-value range checks** (e.g. 16/32-bit) where bit-decomposition
  costs one booleanity product per bit and a real lookup is cheaper.
- **A reusable primitive** so new UAIRs can `declare a table` instead of
  inventing per-constraint residual algebra. The `LookupTableType::{BitPoly,
  Word}` + `chunk_width` API (`uair/src/lookup_types.rs`) already anticipates
  this.

The menu below is aimed primarily at Regime B; M0 is the Regime-A status quo.

---

## 3. Prior art already in this repo (don't re-tread)

| Where | What | Status |
|---|---|---|
| `gkr-logup`, `main-betta-algebraic-lookup`; `piop/src/lookup/gkr_logup/{gkr,protocol,structs,tables}.rs` | **GKR-LogUp, "chunks-in-clear polynomial-valued lift"**: chunks neither committed nor sent; prover sends per-`(ℓ,k)` polynomial-valued chunk lifts `c_k'^(ℓ)=MLE[v_k^(ℓ)](r_inner)∈F_q[X]_{<cw}`; witness-side GKR over `ψ_α`-projected chunks; parent column bound by a **second Zip+ opening at `r_inner`**. | Mature, staged A–D, benched (`BinLookup16`, no-lookup control) — the **incumbent** general lookup. |
| `piop/src/lookup/booleanity.rs`; `sha256.rs` virtual residuals | Booleanity zerocheck `v(v−1)=0` + virtual booleanity residuals — the "algebraic lookup" already shipped (M0). | Shipped on `main-beta`. |
| `main-beta-algebraic-lookup` | Booleanity folded into the CPR sumcheck group. | Perf refactor, not a new method. |
| `origin/f/optimize-booleanity-lookup`, `…/optimize-cache-lookups` | Booleanity round-1 fast path; scalar-projection cache. | Perf tuning. |
| `origin/f/binary-poly-lookup` | Bit-op *virtual columns* (shift/rotate) — **misnamed**; not a multiset lookup. | — |
| `documentation/f2x-sha-todo.md` ~L2287–2731 | The **`F_2` "lookup adder"**: multiplicative grand product over an 8-bit limb add table to replace the 12 trusted carry Hadamards; sound only multiplicatively in char 2. | Analyzed, not implemented (char-2 path). |

Takeaway: **GKR-LogUp is already built**; the value of this branch is a
comparative decision (and possibly Regime-A systematization), not a from-scratch
LogUp.

---

## 4. The methods

For each: the idea/identity, committed-data cost (the resource that matters
here), prover/proof/verifier sketch, soundness, step-4 integration, and when to
reach for it.

### M0 — Algebraic / virtual-column discharge *(status quo; generalize it)*
- **Idea.** Encode the table as algebra the prover already runs:
  - *Booleanity*: `v_k(v_k−1)=0` per bit-slice (degree-3 zerocheck), SZ-recombined.
  - *Virtual booleanity residuals*: declare a linear combination as a virtual
    binary-poly column and let booleanity pin it (the Ch/Maj trick).
  - *Low-degree vanishing / ideal-membership range*: enforce `v∈{0,…,B}` by the
    single zerocheck term `∏_{j=0}^{B}(v−j)=0` (degree `B+1`). For an unpacked
    small carry this is one extra degree-`≤7` term, **zero extra commitment**.
    (Native to UCS: a range is an ideal-membership predicate; `[0,2^w)` is
    `ψ_2` of a degree-`<w` bit-poly.)
- **Cost.** *No extra committed columns.* Adds degree to the relevant sumcheck
  group (booleanity is degree 3; a `∏(v−j)` term raises that group to `B+1`).
- **Soundness.** Schwartz–Zippel on the recombination / vanishing polynomial.
  Characteristic-agnostic.
- **Integration.** Already lives in the step-4 booleanity/CPR groups.
- **Use when.** Tables are tiny and bit-structured (Regime A). **This is the
  right answer for the μ-carries and Ch/Maj** — and the only open work is to
  turn the μ self-enforcement (`sha256.rs:322`) into an explicit declarable
  `∏(v−j)` range check, closing the NYI caveat cheaply.

### M1 — Direct-sumcheck LogUp (Haböck)
- **Identity.** `Σ_i 1/(α−a_i) = Σ_t m_t/(α−T_t)` over `F_q`; prove via a
  sumcheck on the cleared-denominator relation plus inverse well-formedness
  `h·(α−·)=1`.
- **Cost.** Commits the **multiplicity column** `m` *and* helper **inverse
  columns** → several extra Zip+ commitments + openings. Needs batched field
  inversion.
- **Soundness.** Sound over large `F_q`. **Unsound in char 2** (don't port to
  `f2-clean`).
- **Integration.** Extra sumcheck group at step 4; extra committed oracles.
- **Use when.** You want the simplest table-agnostic argument and don't mind the
  committed-column overhead. In *this* PCS that overhead is the weak point —
  which is why M2 usually dominates.

### M2 — LogUp-GKR  *(incumbent; already built)*
- **Idea.** Same log-derivative identity, but evaluate the fractional sum with a
  **layered GKR fraction-addition circuit** (gate:
  `(p₁,q₁)+(p₂,q₂)=(p₁q₂+p₂q₁, q₁q₂)`) instead of a direct sumcheck. The repo
  variant keeps **chunks in clear** (polynomial-valued lifts) and binds the
  parent column with a **second Zip+ opening at `r_inner`**.
- **Cost.** **Commits essentially nothing extra** beyond the unavoidable
  multiplicity data; no helper inverse columns; the fraction tree is
  GKR-discharged. Trades commitment for a few extra GKR rounds (cheap in a
  Brakedown proof that is already large) + one extra opening.
- **Soundness.** Sound over large `F_q`; RBR-KS via the GKR sumchecks.
- **Integration.** Already wired on `gkr-logup`/`main-betta`.
- **Use when.** **Default general primitive for Regime B.** Best fit for the
  commit-dominated cost model.

### M3 — Lasso / Spark  *(decomposable big tables)*
- **Idea.** For a table that decomposes as `T[i₁,…,i_c]=g(T₁[i₁],…,T_c[i_c])`
  (range checks = chunk concatenation; bitwise ops = per-chunk op), prove the
  small sub-table lookups via **Spark** (sparse multilinear commitment) +
  **offline memory checking** (a multiset/grand-product consistency check on
  read counts). Cost `≈ O(#lookups·c + Σ|T_k|)` — independent of the full table
  size.
- **Cost.** Commits **read-counts** and the **sparse sub-table indices**
  (a Spark sparse-commit, i.e. a new Zip+ sparse variant) — heavier machinery.
  The memory-checking grand products are themselves M4-style.
- **Soundness.** Standard (offline memory checking); large-field.
- **Integration.** Decomposition maps directly onto the existing
  `chunk_width` API; needs a sparse-commit path in Zip+.
- **Use when.** Tables are **huge but structured** — ECDSA mod-`n`/big-field
  byte ops, 32/64-bit range checks. Overkill for the SHA carries.

### M4 — Multiplicative grand product / multiset via GKR  *(characteristic-agnostic)*
- **Identity.** Multiset inclusion via `∏_i (α−a_i) = ∏_t (α−T_t)^{m_t}`
  (or the Plookup randomized-difference homogenization), with the products
  discharged by **Thaler's GKR product tree** (no committed running product).
- **Cost.** Commits only multiplicities; products GKR-discharged. Sum trees
  (M2) are marginally cheaper than product trees over large `F_q`.
- **Soundness.** **Sound in *any* characteristic** — `(α−v)²≠0` means even
  multiplicities don't cancel. This is the *only* menu item that works on the
  `f2-clean` path, and it **subsumes the `F_2` "lookup adder"** of
  `f2x-sha-todo.md`.
- **Integration.** GKR groups at step 4 (shared with M2's infrastructure).
- **Use when.** You want **one lookup design for both the integer and `F_2`
  paths**, or you're specifically closing the `F_2` trusted-adder gap.

### Ruled out — KZG-family lookups (cq, Caulk/Caulk+, Baloo)
Tiny proofs / `O(1)` verifier, but require pairing-friendly groups and a
(universal) trusted setup. **Incompatible with Zinc+'s transparent hash-based
commitment** — adopting them abandons transparency. Listed for completeness;
not a candidate.

---

## 5. Decision matrix

| Method | Extra committed cols | Extra opening | Prover | Proof size | Arbitrary tables | Char-2 sound | Repo status |
|---|---|---|---|---|---|---|---|
| **M0** algebraic / virtual | **none** | none | + sumcheck degree | none | bit-structured only | ✅ | **shipped** |
| **M1** direct LogUp | mult. + inverse | yes | cheap sumcheck | + columns | ✅ | ❌ | — |
| **M2** LogUp-GKR | ~none (chunks in clear) | +1 (`r_inner`) | GKR + 2nd open | + GKR rounds | ✅ | ❌ | **built** (sibling) |
| **M3** Lasso/Spark | read-counts + sparse | yes | sub-table sized | + sparse commit | ✅ (huge) | ❌¹ | — |
| **M4** mult. grand product | multiplicities | maybe | GKR product tree | + GKR rounds | ✅ | **✅** | analyzed (`f2` adder) |
| ~~KZG family~~ | n/a | n/a | n/a | tiny | ✅ | n/a | **ruled out** (needs pairings/SRS) |

¹ M3's memory-checking layer can be instantiated multiplicatively (M4) to be
char-2 sound; the Spark sparse-commit is the harder part there.

---

## 6. Recommendation (tiered)

- **Regime A (the SHA-256 carries & Ch/Maj): keep M0.** A multiset lookup is
  not worth it. *Optional, high-value tidy:* promote the μ-carry
  self-enforcement (`sha256.rs:322`) to an explicit declarable range check —
  a single `∏_{j=0}^{B}(v−j)=0` term folded into the booleanity group — to
  close the documented NYI caveat with zero extra commitment.
- **Regime B (general primitive): adopt M2 (GKR-LogUp).** It is already built,
  and its "commit nothing extra, bind via a second opening" shape is the right
  match for the Brakedown cost model. For **huge structured tables** (ECDSA
  mod-`n`, byte ops) layer **M3 (Lasso decomposition)** on top — the
  `chunk_width` API is the hook; the new cost is a Zip+ sparse-commit path.
  Before committing, run an **A/B of M2 vs M1** to quantify the
  committed-column savings on a realistic table.
- **One design for both characteristics: M4 (multiplicative grand product).**
  The only char-2-sound option; it unifies the integer-path primitive with the
  `F_2` "lookup adder."

---

## 7. Concrete first steps on this branch

1. **Lowest-risk, self-contained first commit:** declarable small-range check
   via `∏(v−j)` folded into the step-4 booleanity group; wire it to the μ-carry
   columns and drop the NYI caveat. Pure algebra, no commitment, no new
   subprotocol — a clean way to start `lookup` and exercise the step-4 plumbing.
2. **General primitive:** rebase the `gkr-logup` work onto `lookup`, then A/B
   M2 vs a direct-sumcheck M1 on a realistic table to confirm the
   commit-savings; prototype M3 decomposition for one large table (a 16-bit
   range or an ECDSA byte table) to validate the `chunk_width` path.
3. **If targeting both paths:** prototype M4 on the integer path first (simpler
   to debug over `F_q`), then port to `f2-clean` as the sound lookup adder.

---

## 8. `F_2`-path note

The characteristic-2 lookup direction (multiplicative grand-product "lookup
adder", and *why additive LogUp is unsound there*) is already tracked in
`documentation/f2x-sha-todo.md` (~L2287–2731, L2572). This branch's scope is the
integer path, so that analysis is **cross-referenced, not duplicated** here. If
work on `lookup` ends up touching the `f2_prove.rs` path, add an entry there per
the repo's ledger rule.
