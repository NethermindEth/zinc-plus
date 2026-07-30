# Pointer query (composed reads) — design

Branch: `mariari/pointer-query` (off `main-beta-lookup`). Status: **Stage 1
IMPLEMENTED + e2e-validated** — spec (`ComposedReadSpec`), subprotocol
(`piop/src/pointer_query.rs`), and protocol wiring (`step4c` + two int
openings) all landed; `PointerHopUair` proves e2e, the forged-dereference
twin rejects from `ProtocolError::PointerQuery`, bridge- and lifted-eval
tampers reject, and the full protocol suite stays green. Teeth were grown
first; stages 2-3 (int reducer, region read) remain the menu below.

Goal: a **declarable composed-read check** for the integer path — the verifier-side
"pointer query step" of the FOL paper (Lemma 4.2's `C_i(C_j(X))` obligation): given
a committed value column `V`, committed address-bit columns `b_1..b_mu`, and a
committed result column `R`, enforce

```text
for all x in {0,1}^mu:   R(x) = V(b_1(x), ..., b_mu(x))
```

i.e. every row of `R` is the entry of `V` at the cube position spelled by the bit
columns at that row. This is the primitive that lets a UAIR *dereference its own
trace* — value-addressed reads — without emulating random access inside the
constraint system (the emulation costs `O(len)` committed columns and constraints
per read and is the reason the downstream FOL frontend caps value-addressed
programs at seven steps today).

---

## TL;DR

1. **No lookup argument, no multiplicities, nothing new committed.** The check
   decomposes into one free rider on the step-4 sumcheck plus two chained plain
   sumchecks (`MLSumcheck`), all over MLEs of columns that are already committed.
   This is the M0 "algebraic discharge" spirit of `lookup-methods-design.md`
   applied to reads: the obligation is per-position, not multiset, so the
   log-derivative machinery (M1/M2) is more than it needs.
2. **Cost**: two extra sumchecks of `mu` rounds (degrees 2 and `mu+1`), and — in
   stage 1 — **two extra Zip+ openings** of the int batch, at the two sumcheck
   endpoints. That mirrors exactly how GKR-LogUp first shipped ("a second
   opening, beyond the step-7 one at `r_0`") before `BinMultipointReducer`
   folded its claims; an int-batch reducer that folds `{r_0, r_A, r_B}` into one
   opening is the symmetric future optimization, out of scope here.
3. **Soundness leans on two preconditions the AIR must supply** (both are
   constraints the FOL frontend already emits): the bit columns are boolean
   (CPR booleanity constraints), and pointers land in the meaningful region of
   the cube (a range obligation; see §Range). Given those, the chain below is
   standard Schwartz–Zippel + sumcheck soundness over `F_q`.

---

## The identity, and why it splits in three

Write `eqt(b(x), y) = prod_nu (b_nu(x)*y_nu + (1-b_nu(x))*(1-y_nu))`. On boolean
`b`, `eqt(b(x), ·)` is the indicator of the cube point `b(x)`, so the obligation
is equivalent (after batching rows with `eq(r*, x)`) to:

```text
R~(r*)  =  sum_x eq(r*,x) * sum_y eqt(b(x),y) * V(y)
```

where `r*` is the step-4 shared sumcheck point — already transcript-bound, and
`R~(r*)` is already in the proof as an ordinary resolver up-eval, discharged by
the existing mp-eval → step-7 cascade. So the R side costs **nothing at all** —
no step-4 group, no new bytes — and the component is purely the read side:

- **Phase A — a `y`-sumcheck.** Define `u(y) := sum_x eq(r*,x) * eqt(b(x),y)`;
  on boolean bits this is the eq-mass pushed through the pointer map,
  prover-computable in `O(2^mu)` and never committed. Sumcheck A proves
  `R~(r*) = sum_y u~(y) * V~(y)` (degree 2, `mu` rounds), ending at `r_A` with
  claims `u~(r_A)` and `V~(r_A)`.
- **Phase B — an `x`-sumcheck.** `u~(r_A)` is discharged by proving
  `u~(r_A) = sum_x eq(r*,x) * prod_nu (b_nu(x)*(r_A)_nu +
  (1-b_nu(x))*(1-(r_A)_nu))` — a product of `mu+1` multilinears, so degree
  `mu+1`, `mu` rounds — ending at `r_B` with claims `b~_nu(r_B)`. The identity
  `u~(r_A) = MLE_y[u](r_A)` holds exactly because `eqt(b(x),·)` is multilinear
  in `y` for each cube `x`; booleanity of `b` is what makes `u`'s cube values
  the pushed eq-mass. (This product is, not coincidentally, the same `eqt` the
  downstream frontend's constraint-level emulation unrolls into `len` committed
  columns per read — the component moves it into one sumcheck.)

Multiple reads batch with a transcript challenge `alpha` drawn at step-4c entry:
sumcheck A proves `sum_j alpha^j R~_j(r*)`, one A, one B.

**Endpoint discharge (stage 1):** `V~_j(r_A)` and `b~_(j,nu)(r_B)` are int-batch
column evals at fresh points; two additional `prove_f`/`sample_alphas`+
`verify_with_alphas` pairs at mirrored transcript positions discharge them
against the existing int commitment. No new commitment anywhere.

## Transcript order

Within the standard phase chain:

1. Steps 0–4b run unchanged; step 4's shared rounds fix `r*` and the resolver
   absorbs the up-evals (including every `R~_j(r*)`).
2. `step4c_pointer_query`: draw and absorb `alpha`; sumcheck A runs its rounds;
   the bridge evaluations `u~_j(r_A)` are absorbed; sumcheck B runs its rounds.
3. Step 5 unchanged. Step 6 additionally lifts the witness-int columns at `r_A`
   and `r_B` and absorbs those evaluations after the `r_0` ones.
4. Step 7 additionally opens the int batch at `r_A` then `r_B` (each opening
   draws its own alphas), mirrored exactly in the verifier.

`alpha` and both sumchecks' round challenges are drawn after all commitments
(step 0) and after `r*`, so a malicious prover cannot grind columns against
them; the lifted evaluations at `r_A`/`r_B` are bound by the openings, not by
their absorption position.

## Padding

The check quantifies over the full cube, padding included. The FOL frontend pads
every column with its column-1 value, which makes the identity total: at a
padding row `x`, `b(x) = b(1)`, `R(x) = R(1)`, and the read repeats row 1's.
A UAIR whose padding is not read-consistent must not declare composed reads —
declaration-time validation enforces only shape (below); padding consistency is
the trace generator's obligation, same as for shifts.

## Range (what this component does NOT check)

`R(x) = V(y)` at the cube point `y` spelled by the bits — the *whole cube* is
addressable, padding region included. Whether an address may point at padding is
an AIR-semantics question, so the range obligation `address in the real region`
stays with the AIR, exactly as `lookup-methods-design.md` treats ranges:

- tiny traces: the `prod_j (v - j)` vanishing check (R1/M0), zero commitment;
- general: a region-indicator column read through this same component (declare a
  read of the region column whose expected sum is `sum_x eq(tau,x)*1 = 1`) —
  planned as stage 3, not in this diff;
- the FOL frontend currently emits its own range checks and keeps doing so.

## Declaration surface

```rust
/// One composed read: result_col(x) = value_col at the cube position
/// spelled by bit_cols (low bit first) at x.
pub struct ComposedReadSpec {
    pub value_col: usize,
    pub bit_cols: Vec<usize>,
    pub result_col: usize,
}
```

on `UairSignature` via `with_composed_reads(...)`, validated at construction:
column indices in range, `bit_cols.len() == num_vars`, all three referring to
int columns. (This triple is byte-for-byte the read descriptor the FOL
frontend's Section-4 lowering already produces: `{value_row, bit_rows,
result_row}`.)

## Staging

1. **Stage 1 (this branch):** spec + validation; the two-sumcheck protocol;
   stage-1 discharge via two extra int openings; e2e positive test on a
   hop-style UAIR + tamper-reject suite (forged result entry, forged bridge
   evaluation, forged lifted evaluation). Teeth first: the
   tamper tests are written against the design before the prover exists.
2. **Stage 2:** int-batch multipoint reducer (fold `r_0`/`r_A`/`r_B` to one
   opening), the symmetric move to `BinMultipointReducer`.
3. **Stage 3:** the region-indicator read (declarable range enforcement), and
   the degree-`mu+1` phase-B round cost revisited if profiling warrants
   (an eq-factorized two-phase split is known but not worth the code until a
   trace wall increase makes it so).

## Costs at a glance

| | committed cols | openings | sumcheck rounds | proof bytes |
|---|---|---|---|---|
| constraint-level emulation (today, downstream) | `O(len·mu)` per read | — | — (but `O(len·mu^2)` constraints) | — |
| this component | **0** | +2 (stage 1) → +0 (stage 2) | `mu` (step-4 group, shared) + `2mu` | 2 sumcheck proofs + eval claims |
