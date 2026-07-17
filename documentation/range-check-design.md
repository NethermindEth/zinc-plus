# Z[X] range checks — design

Branch: `lookup` (off `main-beta`, the integer path). Status: design, no code yet.

Goal: a **sound, declarable range-check primitive** for the integer path — assert
each entry of a designated witness quantity lies in `[0, B]` (or a power-of-two
range `[0, 2^w)`). It must work for both the *tiny* ranges already in the SHA-256
UAIR (the μ-carries) and the *larger* limb ranges that big-integer / ECDSA work
needs.

This is the integer-path half of the two-part lookup effort. The `F_2[X]`
"addition mod 2³² via a lookup" half lives on a branch off `f2-clean`; see
`documentation/f2x-sha-todo.md` ("Grand-product LOOKUP adder — implementation
plan", 2026-06-09).

---

## Setting

The integer path projects each witness cell `Z[X] → φ_q → F_q[X] → ψ_α → F_q`
and runs a multi-degree sumcheck at step 4. A range check therefore acts on a
quantity that, post-projection, is a field element `v ∈ F_q` but conceptually a
small non-negative integer. Because `q` is a ~λ-bit prime and the bounds fit
under it (`B < q`), `v ∈ {0,…,B}` *as field elements* ⟺ *as integers* — no
wraparound, so a polynomial-identity check over `F_q` is a faithful integer
range check.

## The Zinc+ exploit: commit lookup data over the projected field F

**The lookup PIOP runs *after* `ψ_α`, entirely over the projected field `F = F_q`.**
So any auxiliary witness a range check introduces — chunk columns, multiplicity
columns, helper columns — can be committed with a **plain commitment over `F`**,
*decoupled from the ring (`Z[X]`) commitment* (Zip+ over IPRS/RAA, with its
integer coefficient-bit-size machinery). Binding to the original witness is a
single **`F`-linear recombination** evaluated at the shared sumcheck point `r*`.

Consequences:
- **The ring commitment is never burdened** by the lookup. Lookup-auxiliary data
  is a *second-class*, cheap `F`-commitment (a vanilla Brakedown over `F_q`), not
  a first-class ring commitment.
- **Chunk width is decoupled from the ring's bit-slice layout** — pick it purely
  to optimize the lookup table size and the `F`-commitment.
- Therefore we are **not limited to commit-free algebra**: chunked lookup range
  checks are cheap and become the natural method for medium/large ranges.

This principle is the backbone of the design below (and of the `F_2` adder).

## Two representations of the quantity

1. **Int column** (a `zinc` integer cell): `v` enters the sumcheck as the column
   MLE evaluated at `r*`.
2. **Bit-packed value** (the μ-carries: `μ_e` = bits 5–7 of the `W_MU_PACKED`
   `binary_poly` column): booleanity already extracts/binds the per-bit slices
   `v_b`; the quantity is the (virtual) linear combination `v = Σ_b 2^b v_b`.

---

## Methods

### R1 — Vanishing-polynomial zerocheck *(tiny `B`; zero commitment)*
Enforce `v ∈ {0,…,B}` by the single identity
`∏_{j=0}^{B}(v − j) = 0` — a degree-`(B+1)` zerocheck over the column / slice
combo. **No commitment at all** (rides the existing sumcheck). Best for `B ≤
~16`. **Exactly fits the μ-carries** (`μ_W∈[0,3]`→deg4, `μ_e∈[0,5]`→deg6,
`μ_a∈[0,6]`→deg7), applied to the linear combination of booleanity slices for
each packed sub-field. It is the `B>1` generalization of the existing
virtual-booleanity-residual trick (`v(v−1)=0` is the `B=1` case,
`sha256.rs:558-648`).

### R2 — Bit-decomposition + booleanity *(power-of-two `[0,2^w)`; zero commitment)*
`v = Σ_{i<w} 2^i b_i`, each `b_i` boolean via the existing booleanity sumcheck,
plus the linear recombination. Free when the bits are virtual slices of an
already-committed `binary_poly` column (booleanity runs anyway).

### R3 — Chunked lookup, chunks committed over `F` *(medium / large `B`; the exploit)*
For ranges too large for R1's degree and where R2's per-bit cost is wasteful
(e.g. 16/32/64-bit limbs in big-integer / ECDSA work):

1. Decompose `v = Σ_{i<k} c_i · 2^{c·i}` into `k` chunks of `c` bits.
2. **Commit the chunk columns `{c_i}` over `F_q`** (vanilla `F`-commitment).
3. **Range-check each chunk** `c_i ∈ [0,2^c)` by a LogUp into the public table
   `{0,…,2^c−1}` over `F_q`; **commit the multiplicity column over `F_q`** too.
   (For small `c` the table is tiny; LogUp is sound over the large field `F_q`.)
4. **Bind** the chunks to the witness with the recombination
   `v = Σ_i c_i 2^{c·i}`, an `F_q`-linear constraint checked at `r*` (where `v`
   is the original ring column's eval). Open the `F`-side chunk/multiplicity
   commitment at `r*` alongside the ring opening.

- **Soundness.** Chunk range checks (LogUp / or recursively R1 on each tiny
  chunk) + recombination (SZ) + uniqueness: with `c_i ∈ [0,2^c)` and
  `Σ < 2^{ck} ≤ q`, the decomposition is unique, so equality in `F_q` forces the
  integer range. (LogUp here is sound because we are over `F_q`, not char 2.)
- **Cost.** Only cheap `F_q` commitments (chunks + one small multiplicity
  column), **no ring-commitment burden** — this is the exploit. A 32-bit range
  check is a handful of `F_q`-committed chunk columns + a tiny LogUp.
- Note R3 can use **R1 instead of LogUp** on each chunk (e.g. `c=8` → degree-256
  vanishing is too high, but `c=4` → degree-16 is fine and avoids the
  multiplicity column entirely). The chunk width trades vanishing-degree vs
  table/multiplicity commitment.

**Rule of thumb:** tiny `B` → **R1** (zero commitment); power-of-two or
already-`binary_poly` → **R2** (zero commitment); medium/large `B` → **R3**
(cheap `F`-side chunk + multiplicity commitments).

---

## Declarable primitive + step-4 integration

- **Declaration.** Reuse/extend the existing API: `LookupTableType::Word {
  width }` (`uair/src/lookup_types.rs`) = `[0,2^width)` → R2 or R3; add a `Range
  { lo, hi }` variant → R1 (small) or R3 (large). A spec names the target: a
  whole column (case 1) or a `(column, bit-range)` sub-field (case 2).
- **Prover.** In `protocol/src/prover.rs::step4_sumcheck`, add a range group
  beside CPR and booleanity: the degree-`(B+1)` `∏(v−j)` poly (R1), reuse of
  booleanity slices (R2), or — for R3 — build the chunk MLEs, commit them +
  multiplicities over `F` (new flat-`F` PCS instance, sibling to the existing
  per-type ones), and feed the chunk-range LogUp + recombination into
  `MultiDegreeSumcheck::prove_as_subprotocol` sharing `r*`.
- **Verifier.** The range group's claimed sum is `0` (zerocheck); the verifier
  recomputes the expected eval from the column / slice / chunk evals at `r*`. For
  R3 it also opens the `F`-side commitment at `r*` and checks the recombination.
- **What's new plumbing-wise:** only R3 adds anything — a **flat-`F` commitment
  path** for chunk/multiplicity columns (a 4th PCS instance over `F_q`, alongside
  binary_poly/arbitrary_poly/int) and its opening at `r*`. R1/R2 are pure
  additions to the existing sumcheck.

## What it closes / enables

- The μ-carry **NYI range-check caveat** (`sha256.rs:115-120`) and the
  **self-enforcement** argument (`:322-328`) become a direct, declarable R1
  check.
- A general, sound **`[0,B]` / `[0,2^w)` range-check capability** any
  integer-path UAIR can declare — including **large limb ranges** (via R3) at the
  cost of only cheap `F`-side commitments.

## Suggested implementation order

1. **R1** (vanishing-poly group) in step 4 — smallest, self-contained, zero new
   commitment; validate on the μ-carries (`μ_W,μ_a,μ_e`). Foundational: exercises
   the step-4 range-group plumbing.
2. **R2** routing for `Word{width}` (mostly wiring to existing booleanity).
3. **R3** — the chunked-over-`F` path: stand up the flat-`F` commitment instance
   + opening, the chunk-range LogUp, and the recombination binding. This is the
   piece that uses the exploit and unlocks large ranges.
4. e2e test: an out-of-range witness must fail; honest passes.
