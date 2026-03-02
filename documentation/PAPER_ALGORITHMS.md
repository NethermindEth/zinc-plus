# Zinc+ Paper: Complete Algorithmic Specification

> Extracted from the LaTeX source at `/Users/albertgarretafontelles/dev/zinc-/`.
> Purpose: serve as a ground-truth reference to compare against the Rust implementation in `zinc-plus-new`.

---

## Table of Contents

1. [UCS – Universal Constraint System](#1-ucs--universal-constraint-system)
2. [UAIR+ Relation](#2-uair-relation)
3. [Zinc+ PIOR Compiler (Protocol Π^UCS)](#3-zinc-pior-compiler)
4. [AirZinc – Full IOP for UAIR+](#4-airzinc--full-iop-for-uair)
5. [Projected PIOP (R_UAIR,proj)](#5-projected-piop)
6. [Ideal Check](#6-ideal-check)
7. [Projections (Ring → Field)](#7-projections)
8. [Sumcheck Protocol](#8-sumcheck-protocol)
9. [Combined Polynomial Resolver (CPR)](#9-combined-polynomial-resolver)
10. [Batched Shift PIOP](#10-batched-shift-piop)
11. [LogUp PIOP (Lookup)](#11-logup-piop)
12. [Decomposition+LogUp PIOP](#12-decompositionlogup-piop)
13. [Zip+ IOPP / PCS](#13-zip-iopp--pcs)
14. [IPRS Codes](#14-iprs-codes)
15. [Compilation (Poly-IOP → IOP)](#15-compilation)
16. [SHA-256 Arithmetization](#16-sha-256-arithmetization)
17. [ECDSA Arithmetization](#17-ecdsa-arithmetization)
18. [Implementation Mapping](#18-implementation-mapping)

---

## 1. UCS – Universal Constraint System

**Source:** `UCS/general_definition.tex` (Definition `def:ucs`)

### Index

```
idx = (witlen, inplen, N_rows, (q_1,...,q_P), B, (d_0,...,d_P), constraints)
```

| Field | Meaning |
|-------|---------|
| `witlen` | witness length (number of columns) |
| `inplen` | public input length |
| `N_rows` | number of rows |
| `(q_1,...,q_P)` | tuple of prime powers |
| `B` | bit-size bound for Q[X] coefficients |
| `(d_0,...,d_P)` | degree bounds (d_0 for Q[X], d_i for F_{q_i}[X]) |
| `constraints` | family of (Q_{ij}, I_{ij}) pairs |

### Constraints Structure

```
constraints = ( (Q_{0j}, I_{0j})_{j∈[J]}, (Q_{ij}, I_{ij})_{i∈[P], j∈[J]} )
```

Where:
- `Q_{0j} ∈ Q^{<B,d_0}[X][r, y, Z_1]` — Q[X]-constraint polynomial
- `Q_{ij} ∈ F_{q_i}[X][r, y, Z_1, Z_2]` — F_{q_i}[X]-constraint polynomial
- `I_{0j} = (g_{0j})` — ideal in Q[X] given by generator g_{0j}
- `I_{ij} = (g_{ij})` — ideal in F_{q_i}[X] given by generator g_{ij}

### Witness

```
wit = (f_0, f_1, ..., f_P)
```

- `f_0 ∈ (Q^{<B,d_0}[X])^{witlen}` — integer polynomial witness
- `f_i ∈ (F_{q_i}^{<d_i}[X])^{witlen}` — finite field polynomial witness

### Input

```
inp = y ∈ (Z_{(q_1···q_P)}^{<B,d_0}[X])^{inplen}
```

(Entries in the intersection localization ring.)

### Satisfaction Conditions

**(i) Q[X]-constraints:** For every row `r ∈ [N_rows]` and constraint `j ∈ [J]`:
```
Q_{0j}(r, y, f_0) ∈ I_{0j} ⊆ Q[X]
```

**(ii) F_{q_i}[X]-constraints:** For every `i ∈ [P]`, `r ∈ [N_rows]`, `j ∈ [J]`:
```
Q_{ij}(r, φ_{q_i}(y), φ_{q_i}(f_0), f_i) ∈ I_{ij} ⊆ F_{q_i}[X]
```

**(iii) Well-definedness of φ_{q_i}:**
```
f_0 ∈ (Z_{(q_1···q_P)}^{<B,d_0}[X])^{witlen}
```

### Lookup Constraints in UCS

**Source:** `UCS/structural_instantiations.tex`

Lookup `w_i ∈ {t_j}` is encoded as a zero-test of:
```
Q_Look(r, Z) = ∏_{j∈[T]} (L(r,Z) - t_j)
```
where `L(r, Z) = Σ_k Z_k · ℓ_{[N],k}(r)` is the Lagrange selector.

> In practice, the high-degree polynomial identity is NOT used; instead LogUp is invoked directly.

---

## 2. UAIR+ Relation

**Source:** `hybrid_arithmetics/UAIR.tex` (Definition `def:uairplus`)

UAIR+ is the trace-oriented specialization of UCS.

### Index

```
idx = (N_rows, N_cols, (q_1,...,q_P), B, (d_0,...,d_P), constraints, lookupsets)
```

### Witness = Traces

```
f_0 ∈ (Q^{<B,d_0}[X])^{N_rows × N_cols}     — Q[X]-trace
f_i ∈ (F_{q_i}^{<d_i}[X])^{N_rows × N_cols}  — F_{q_i}[X]-trace (for i ∈ [P])
```

### Constraints

Applied **uniformly** across all rows:

**(i) Trace-local Q[X]-constraints:** For every row r ∈ [N_rows], j ∈ [J]:
```
Q_{0j}(y[r], f_0[r], f_0↓[r]) ∈ I_{0j}
```
where `f_0↓[r]` denotes shifted-row access (row r+1, clamped at last row).

**(ii) Trace-local F_{q_i}[X]-constraints:** Similarly for each prime i.

**(iii) Column-wise lookups:** For each column c with a lookup set L_{ic}:
```
set({f_{i,c}[r] : r ∈ [N_rows]}) ⊆ L_{ic}
```

### Extensions

- **Shift access:** Constraints can reference `f[r]`, `f↓[r]` (row r+1), and `f↓↓[r]` (row r+2)
- **Lookup on algebraic combinations:** Lookups can be on polynomial functions of column entries

---

## 3. Zinc+ PIOR Compiler

**Source:** `PIOR/pior_compiler.tex` (Protocol Π^UCS), `crypto/crypto_piop.tex`

The compiler reduces UCS over mixed rings to a family of projected algebraic relations over finite fields.

### Three-Step Process (Conceptual)

#### Step 1: Prime Projection (Q[X] → F_q[X])
- V samples an Ω(λ)-bit prime `q_0 ∈ primeset`
- Replace Q[X] constraints with `φ_{q_0}`-projected constraints in `F_{q_0}[X]`
- Soundness: by prime projection lemma

#### Step 2: Ideal Check via Random MLE Evaluation
- V samples `r ∈ F_q^μ`
- P sends `e = ṽ(r) = Σ_b Q(b, φ_q(y), φ_q(f_0)) · eq(b; r)`
- V checks `e ∈ I'` (projected ideal)
- Reduces to: `Σ^{r,e} = 0` in `F_q[X]`

#### Step 3: Evaluation Projection (F_q[X] → F_q)
- V samples `a ∈ F_q` (or extension)
- Both evaluate at `X = a`: `ψ_{q,a}(Σ^{r,e})(ψ_{q,a}(y), ψ_{q,a}(f_0)) = 0`
- Reduces to a pure finite-field claim

### Formal Protocol Π^UCS

1. **P sends** polynomial oracles `õ(f̃_0), ..., õ(f̃_P)`
2. **V sends** random prime `q_0 ∈ primeset`, random points `r_0 ∈ F_{q_0}^μ`, `r_i ∈ F_{q̃_i}^μ`
3. **Abort check** if constraints/data not in localization ring
4. **P sends** `e_{0j}` and `e_{ij}` — MLE evaluations of constraint images (ideal check values)
5. **V rejects** if `e` values not in correct ideals — **IDEAL MEMBERSHIP CHECK**
6. **V sends** projection points `a = (a_0, ..., a_P)` sampled from `F_{q̃_i}`
7–11. **For each i:** compute projected constraints `Q'_{ij}`, projected inputs `y'_i`, projected oracles `f̃'_{i0}`, `f̃'_{i1}`; V outputs `(idx'_i, inp'_i)`; P outputs `wit'_i`

---

## 4. AirZinc – Full IOP for UAIR+

**Source:** `concrete_snarks/rolled_out_argument.tex` (Algorithm)

This is the **complete rolled-out protocol** instantiating Zinc+ for UAIR+.

### Phase 1: Witness Commitment

1. **Encode & commit:** For each trace `f_i`, encode witness columns via code `C_i`, send codeword oracles
2. **Prime sampling:** V samples `q_0 ∈ primeset`
3. **Out-of-domain sampling:** V samples random points `z_{i,wit} ∈ R_i^μ` for each `i ∈ {0} ∪ [P]`
4. **Out-of-domain evaluations:** P sends evaluations `w_{i,j}` of each oracle at `z_{i,wit}`. Initialize claim list `L`

### Phase 2: Zinc+ Compiler

5. **MLE evaluation points:** V samples `r_i ∈ F_{q̃_i}^μ` for ideal check
6. **MLE evaluations:** P sends `e_{i,j}` = MLE evaluations of constraint images
7. **Ideal membership check:** V checks each `e_{i,j} ∈ I'_{ij}` (rejects if any fail)
8. **Projection points:** V samples `a_i ∈ F_{q̃_i}` for evaluation projection

### Phase 3: Projected PIOP (for each i)

9. **Constraint batching:** V samples random `α ∈ F_{q̃_i}` for batching all projected constraints
10. **μ-round SUMCHECK:** For batched constraint sum `Σ_{r∈{0,1}^μ} batched_constraint(r) · eq(r, ρ) = 0`
11. **Oracle evaluations at sumcheck point** via **lift-and-project**:
    - Lift point `ρ_schk` via `ψ'^{-1}` to polynomial ring
    - P sends claimed evaluation `ū` in polynomial ring
    - V projects: `u = ψ'(ū)` and checks consistency
    - Record claim in list `L`
12. **Batched shift protocol** (`prot:batched_shift`): Reduce shifted-row evaluation claims to new evaluation claims for original oracles
13. **Lookup via LogUp** (`prot:logup`): For each column with nontrivial lookup set, run LogUp (or Decomposition+LogUp for large tables)

### Phase 4: Oracle Consistency via IOPP

14. **Batched IOPP** for each code `C_i`: Run Π_BMLE to verify all accumulated evaluation claims in `L`

### Phase 5: Decision

15. **Accept** iff: (1) all ideal membership checks pass, (2) all sumcheck verifiers pass, (3) all IOPP verifiers pass

### Lift-and-Project Procedure (Algorithm)

```
Input: evaluation point ρ ∈ F_{q̃_i}, oracle index j, code C_i
1. Lift: compute ρ̄ = ψ'^{-1}(ρ) ∈ F_{q_i}^{<ℓ}[X]^μ
2. P sends claimed ū = f̃_{i,j}(ρ̄) ∈ F_{q_i}^{<d_i}[X]
3. V projects: u = ψ'(ū) ∈ F_{q̃_i}
4. Add claim (oracle f̃_{i,j}, point ρ̄, value ū, code C_i) to list L
```

---

## 5. Projected PIOP

**Source:** `concrete_snarks/piop_uair_instantiation.tex`

The Projected UAIR+ relation `R_{UAIR,proj}(q)` is a single-field version of UAIR+.

### Three PIOP Components

1. **Constraint batching + sumcheck** — batch all projected constraints with random coefficients, then run standard sumcheck
2. **Shift reduction** — reduce shifted-row evaluation claims via `prot:batched_shift`
3. **Lookup via LogUp** — prove column-wise lookup constraints via LogUp or Decomposition+LogUp

### Algebraic constraints (as MLE sum)

```
(*) Σ_{r∈{0,1}^μ} Σ_j α_j · Q'_{ij}(y'[r], f'_{i0}[r], f'_{i0}↓[r], f'_{i1}[r], f'_{i1}↓[r]) · eq(r, ρ) = 0
```

### Lookup constraints

```
(†) set({ψ'_i(f_{i,c})[r] : r ∈ [N]}) ⊆ ψ'_i(L_{ic})    for each column c
```

---

## 6. Ideal Check

**Source:** `PIOR/pior_ideal_membership.tex`, `crypto/crypto_piop.tex`

### Ideal Batching Lemma

If `v ∈ K^{<d}[X]^{2^μ}` and `v(x_0) ∉ I` for some `x_0 ∈ {0,1}^μ`, then:
```
Pr_{r∈S^μ}[ṽ(r) ∈ I] ≤ μ / |S|
```

Where `ṽ` is the MLE of `v`, and `S` is the challenge set.

### How It's Used

For each constraint `Q_{ij}` with ideal `I_{ij}`:
1. P computes `e_{ij} = Σ_{b∈{0,1}^μ} Q_{ij}(b, y, f) · eq(b; r)` (the MLE of the constraint image at random point r)
2. V checks `e_{ij} ∈ I_{ij}` (ideal membership)
3. If any check fails, V rejects (soundness from the batching lemma above)

### MLE Commutes with Projection

Key lemma: `ψ(ṽ(ψ^{-1}(u))) = ψ̃(v)(u)` — MLE evaluation commutes with the evaluation projection, enabling the lift-and-project technique.

---

## 7. Projections

**Source:** `PIOR/pior_ideal_membership.tex`, `crypto/crypto_piop.tex`

### Prime Projection φ_q

```
φ_q : Q[X] → F_q[X]
```
Reduces integer polynomial coefficients modulo prime q.

### Evaluation Projection ψ_{q,a}

```
ψ_{q,a} : F_q[X] → F_{q̃}     via f(X) ↦ embedding(f)(a)
```

Where `q̃ = q^ℓ` for sufficiently large extension degree ℓ.

### Right-Inverse (Lifting)

```
ψ_{q,a}^{-1} : F_{q̃} → F_q^{<ℓ}[X]
```

Used in the lift-and-project procedure to lift finite-field evaluations back to polynomial evaluations.

### Extend Map

```
extend: PP → PP     mapping q → q^ℓ
```

Where ℓ is chosen so that `|F_{q^ℓ}|` is large enough for security.

---

## 8. Sumcheck Protocol

**Source:** `IOPP/batched_iopp_multilinear.tex` (steps 2–5), `concrete_snarks/rolled_out_argument.tex` (step 10)

### Standard μ-Round Sumcheck

**Claim:** `Σ_{x∈{0,1}^μ} h(x) = s_0`

**Round i = 1, ..., μ:**
1. P sends univariate `g_i(X)` such that `g_i(0) + g_i(1) = s_{i-1}`
2. V samples `r_i ← F`
3. V sets `s_i = g_i(r_i)`

**Final check:** At the end, `s_μ` must equal `h(r_1, ..., r_μ)`, which is verified against oracle evaluation(s).

### Usage in AirZinc

The sumcheck is used in multiple places:
- **Constraint batching** (step 10): sum over `{0,1}^μ` of batched projected constraints
- **Shift reduction** (step 12): reduce shifted-MLE evaluations to regular MLE evaluations
- **LogUp** (step 13): verify log-derivative identity
- **Zip+ IOPP** (step 14): batched MLE evaluation claims → inner product constraints

---

## 9. Combined Polynomial Resolver (CPR)

**Source:** Implicitly in `crypto/crypto_piop.tex` and `concrete_snarks/rolled_out_argument.tex`

The CPR combines the results of multiple constraint polynomial evaluations into a single batched check.

### Role in Protocol

After the evaluation projection (Step 3), the prover and verifier need to verify:
```
Σ_j α_j · Q'_{ij}(y', f'_0, f'_0↓, f'_1, f'_1↓) = 0     for all rows r
```

The CPR:
1. **Batches** all J constraints with random coefficients `α_1, ..., α_J`
2. **Combines** with the eq polynomial: `Σ_r h(r) · eq(r, ρ) = s` where `h(r) = Σ_j α_j · Q'_j(r, ...)`
3. **Feeds** the result into the sumcheck protocol

### Implementation Note

In the implementation, CPR corresponds to `piop/src/combined_poly_resolver.rs` which handles the multi-constraint batching and evaluation.

---

## 10. Batched Shift PIOP

**Source:** `Appendix/[not included] gadget PIOP's on finite fields.tex` (Protocol `prot:batched_shift`)

### Forward Shift Definition

One-step: `v↓[r] = v[r+1]` for `r < N-1`, `v↓[N-1] = v[N-1]` (clamp).

Two-step: `v↓↓[r] = v[r+2]` for `r < N-2`, clamp last two entries.

### Shifted MLE as Virtual Oracle

The MLE of the shifted vector satisfies:
```
ṽ↓(r) = Σ_{b∈{0,1}^μ} shiftdown(r, b) · ṽ(b)
```

Where `shiftdown(r, b) = next̃(r, b) + eq((r,b), (1,1))` and `next(x, y) = 1 iff y = x + 1`.

For two-step shift:
```
shiftdown2(r, b) = next2̃(r, b) + eq((r,b), (1,1)) + eq((r,b), (1-e_μ, 1))
```

### Batched Shift Protocol

**Relation:** Given J oracles `{ṽ_j}`, evaluation point `r`, claimed values `{y_j}`, shift types `{shift_j}`:
```
y_j = Σ_{b∈{0,1}^μ} shift_j(r, b) · ṽ_j(b)     for all j ∈ [J]
```

**Protocol:**
1. V samples batching coefficients `γ_1, ..., γ_J ← F`
2. P and V run sumcheck for:
   ```
   Σ_{j∈[J]} γ_j · y_j = Σ_{b∈{0,1}^μ} Σ_{j∈[J]} γ_j · shift_j(r, b) · ṽ_j(b)
   ```
3. Sumcheck reduces to openings at a single random point `ρ ∈ F^μ`
4. V queries each oracle `ṽ_j` at `ρ` to obtain `ṽ_j(ρ)`
5. V checks sumcheck consistency and accepts

---

## 11. LogUp PIOP

**Source:** `Appendix/[not included] gadget PIOP's on finite fields.tex` (Protocol `prot:logup`)

### Lookup Relation

```
R_lookup = { (idx, ∅; w) | idx = (N, T, tab, K), w ∈ K^N, w_i ∈ {tab_j : j ∈ [T]} for all i }
```

### Protocol

1. **P computes** multiplicity vector `m ∈ K^T` where `m_j` = count of `tab_j` in `w`; sends oracle `õ(m̃)`
2. **V samples** random challenge `β ← K`
3. **P and V verify** the log-derivative identity:
   ```
   Σ_{i∈[N]} 1/(β - w_i) = Σ_{j∈[T]} m_j/(β - tab_j)
   ```
   via sumcheck (two methods below)
4. **P and V** run sumcheck to verify `Σ_{j∈[T]} m_j = N`
5. **V accepts** iff all sumcheck verifiers accept

### Methods for Proving the Log-Derivative Identity

**Inverse-commitment variant:**
- P sends oracles for `u_i = 1/(β - w_i)` and `v_j = 1/(β - tab_j)`
- Correctness via sumchecks: `(β - w_i)·u_i = 1` for all i, `(β - tab_j)·v_j = 1` for all j
- Final sumcheck: `Σ u_i - Σ m_j·v_j = 0`

**GKR-style variant:**
- Evaluate the rational expression via layered circuit + GKR-style sumcheck
- Avoids commitment to u and v oracles

---

## 12. Decomposition+LogUp PIOP

**Source:** `Appendix/[not included] gadget PIOP's on finite fields.tex` (Protocol `prot:decomplogup`)

For large tables that admit a low-degree algebraic decomposition.

### Table Decomposition Definition

Table `T ∈ F^{|T|}` admits decomposition w.r.t. `(T_1, ..., T_α)` and multilinear `g` if:
```
T(y) = g(T_1(y_1), ..., T_k(y_1), T_{k+1}(y_2), ..., T_α(y_c))
```
for all `y = (y_1, ..., y_c) ∈ {0,1}^μ`.

### Protocol

1. **Table decomposition:** P computes chunk vectors `w_1, ..., w_α` and sends their MLE oracles
2. **Decomposition consistency sumcheck:** Verify via sumcheck:
   ```
   Σ_{y∈{0,1}^μ} (w̃(y) - g(w̃_1(y_1), ..., w̃_α(y_c))) · eq(y, r) = 0
   ```
3. **Lookup step:** For each chunk `i ∈ [α]`, run LogUp with index `(N, |T|^{1/c}, T_i, K)` and witness `w_i`
4. **V accepts** iff decomposition consistency and all LogUp invocations accept

---

## 13. Zip+ IOPP / PCS

**Source:** `IOPP/batched_iopp_multilinear.tex` (Protocol Π_BMLE), `IOPP/iopp_constrained_interleaved.tex` (Protocol Π_CIC), `crypto/crypto_iopp.tex`

### 13.1 Batched IOPP with Multilinear Evaluation Constraints (Π_BMLE)

This is the **full Zip+ PCS protocol**.

**Parameters:**
- Code `M ∈ Q^{n×k_1}` (or `F_q^{n×k_1}`)
- Dimensions `k_1, k_2` (where `μ = k_1 + k_2` is number of MLE variables)
- `J` oracles, `T` MLE constraints
- Degree `d`, bit-size `B`, modulus `m`, proximity `β`

**Protocol Steps:**

1. **Batching:** V chooses batching coefficients `γ_t, γ'_j`; sets initial sum:
   ```
   s_0 = Σ_j γ'_j · Σ_t γ_t · α_{t,j}
   ```

2-5. **μ-round SUMCHECK:**
   ```
   Round i: P sends g_i(X) with g_i(0) + g_i(1) = s_{i-1}
            V samples r_i, sets s_i = g_i(r_i)
   ```

6. **Sumcheck final check:** `s_μ = Σ_j γ'_j · Σ_t γ_t · w̃_j(z') · eq(z_{t,j}, z')`
   where `z' = (r_1, ..., r_μ)`

7. **P sends** claimed evaluations `α'_j = w̃_j(z')` for each oracle j

8. **Decompose & reindex:** Decompose polynomial coefficients based on k_1 × k_2 split

9. **Random linear combination:** V chooses `ζ` coefficients across `J'` oracles

10. **P responds** with `b_sumcheck` satisfying inner product constraint

11. **Run Π_CIC** on derived instance

### 13.2 Inner-Product Constrained Interleaved Code IOPP (Π_CIC)

**Base layer of Zip+.**

**Protocol Steps:**

1. V chooses `r_1, ..., r_J ∈ [2^K]`, sends to P. Set `v* = Σ r_j · v_j`
2. P responds with `w ∈ (Q^{<B})^k` satisfying `w · u*_t = α*_t` for all t
3. V chooses C spot-check indices, verifies `(v*)_{ℓ_i} = (Mw)_{ℓ_i}`

**Soundness:**
```
err_1 = (J-1) · err_pg + L / 2^{K-1}
err_2 = (1 - β)^C
```

### 13.3 Key Reductions in Zip+ (Overview)

1. **Polynomial ring → base ring:** Decompose `V ∈ (Q^{<d}[X])^n` into `d` coefficient matrices. V samples random coefficients to combine
2. **MLE → inner products:** `w̃'(z) = α'` becomes bilinear form `q_2^T · w' · q_1 = α'`. P provides intermediate `b`
3. **Random linear combination across oracles:** V chooses `r_1, ..., r_{k_2}`, P sends `w* = Σ r_j · w'_j`. Spot checks follow

---

## 14. IPRS Codes

**Source:** `crypto/crypto_iprs.tex`

### Definition

IPRS (Integer Pseudo Reed-Solomon) codes lift FFT encoding from F_q to Z:
- Same butterfly structure as FFT over F_q
- Integer arithmetic — **no modular reduction**
- Encoding: `Enc_IPRS : Z^k → Z^n`

### Key Properties

| Property | Value |
|----------|-------|
| Distance | `δ = 1 - k/n + 1/n` (MDS-optimal) |
| Norm bound | `‖Enc(x)‖ ≤ ‖x‖ · (q/2)^{depth+1} · k` |
| Centered reps | ~20% reduction in practical norm growth |

### Performance (Benchmarks from Paper)

| Setting | Encoding Time |
|---------|---------------|
| 2^10 messages in {0,...,2^32-1} | 72.4 μs |
| 2^10 messages in BitPolyset{32} | 552.3 μs |
| SHA-256 (16 compressions) | ~5.89 ms total |

---

## 15. Compilation (Poly-IOP → IOP)

**Source:** `compilation.tex` (Theorem `thm:compilation`, Algorithm `alg:compiled_iop`)

### Ingredients

1. **Poly-IOP** `Π_Poly` for relation R with round-by-round knowledge soundness
2. **Error-correcting codes** `C_qs` for each field (Q and F_{q_i})
3. **Batched IOPPs** `Π_Prox,qs` with multilinear evaluation constraints

### Compiled IOP Protocol

**Round 1 (Commitment):**
- P sends encoded codewords `v_{qs,s}` for each field `qs` and oracle `s`
- V samples out-of-domain points `z_qs` and prime `m`

**Round 2 (Out-of-domain evaluations):**
- P sends evaluations `α_{qs,s}` at each `z_qs` (mod `m` for Q case)

**Rounds 3..R (Poly-IOP simulation):**
- V generates query sets `S^{(i)}_{qs,s}` and sends verifier messages
- P responds with oracle evaluations at query points
- V simulates the Poly-IOP verifier

**IOPP phase:**
- For each field `qs`, run batched IOPP `Π_Prox,qs` with:
  - Oracles: `v_{qs,s}` for each s
  - Code: `C_qs`
  - Constraints: `z → α` for all queried points and responses

### Complexity

| Metric | Value |
|--------|-------|
| Round complexity | `1 + R_poly + R_prox` |
| Query complexity | `qu_poly + Σ_qs (s_qs + qu_prox(s_qs, k_{2,qs}))` |

### Fiat–Shamir

By RBR knowledge soundness + [BCFW25], the compiled IOP can be made non-interactive in the ROM via Fiat–Shamir.

---

## 16. SHA-256 Arithmetization

**Source:** `hybrid_arithmetics/sha256/sha256_intro.tex`, `sha_256_compression_loop_lemmas.tex`, `sha256_message_schedule_lemmas.tex`, `sha256_arithmetization.tex`

### UAIR+ Configuration

| Parameter | Value |
|-----------|-------|
| prime tuple | `q = (2)`, so P = 1 |
| N_rows | 65 (rounds 1–64 + final state) |
| Q[X]-trace columns | 13 (10 bit-poly + 3 integer carries) |
| F_2[X]-trace columns | 4 (shift quotients + remainders) |

### Q[X]-Trace Columns (13)

| Columns | Type | Description |
|---------|------|-------------|
| â, ê, Ŵ | BitPoly{32} | working variables + message word |
| Σ̂₀, Σ̂₁ | BitPoly{32} | Sigma function outputs |
| M̂aj | BitPoly{32} | Majority output |
| t̂_{ef}, t̂_{¬e,g} | BitPoly{32} | Ch helper products |
| σ̂₀, σ̂₁ | BitPoly{32} | message schedule outputs |
| μ_a, μ_e, μ_W | Integer | carry for modular additions |

### F_2[X]-Trace Columns (4)

| Columns | Description |
|---------|-------------|
| S₀, S₁ | shift quotients for σ₀, σ₁ |
| R₀, R₁ | shift remainders for σ₀, σ₁ |

### Constraint Formulas

#### Rotation (ROTR)

```
ROTR^r(u) - X^{w-r} · u ∈ (X^w - 1) in R[X]
```

#### Σ₀ (uses rotations 2, 13, 22)

```
φ₂(â) · ρ₀(X) - φ₂(ŷ_{Σ₀}) ∈ (X^w - 1)_{F₂}
where ρ₀(X) = X^{w-2} + X^{w-13} + X^{w-22}
```

#### Σ₁ (uses rotations 6, 11, 25)

```
φ₂(ê) · ρ₁(X) - φ₂(ŷ_{Σ₁}) ∈ (X^w - 1)_{F₂}
where ρ₁(X) = X^{w-6} + X^{w-11} + X^{w-25}
```

#### Ch (choice function)

```
ê + f̂ - 2·t̂_{ef} ∈ BitPolyset{w}          AND
(1_w - ê) + ĝ - 2·t̂_{¬e,g} ∈ BitPolyset{w}
```

#### Maj (majority function)

```
â + b̂ + ĉ - 2·m̂ ∈ BitPolyset{w}     (parity-majority decomposition)
```

#### Modular Addition (via (X-2)_Q bridge)

```
ŷ - Σ x_j - Σ ẑ_j + μ · X^w ∈ (X-2)_Q
```

Where:
- `ŷ` is the output bit-polynomial
- `x_j` are the bit-polynomial inputs
- `ẑ_j` are auxiliary bit-polynomial terms
- `μ` is an integer carry

#### σ₀ (message schedule, uses rotations 7, 18 + right shift 3)

```
φ₂(x̂) · ρ_{σ₀}(X) + S₀ - φ₂(ŷ_{σ₀}) ∈ (X^w - 1)_{F₂}
where ρ_{σ₀} = X^{w-7} + X^{w-18}
```

Right shift: `x̂ = R + X^r · S` where `S = SHR^r(x)`

#### σ₁ (message schedule, uses rotations 17, 19 + right shift 10)

```
φ₂(x̂) · ρ_{σ₁}(X) + S₁ - φ₂(ŷ_{σ₁}) ∈ (X^w - 1)_{F₂}
where ρ_{σ₁} = X^{w-17} + X^{w-19}
```

### Shift-Register Structure

The following are simple copy constraints (shift access):
```
b_{t+1} = a_t,   c_{t+1} = b_t,   d_{t+1} = c_t
f_{t+1} = e_t,   g_{t+1} = f_t,   h_{t+1} = g_t
```

These are handled via the shift access in UAIR+ constraints (referencing row r and row r+1).

### Public Input

10 columns: 8 state registers at rows 1 and 65, message words, round constants.

---

## 17. ECDSA Arithmetization

**Source:** `hybrid_arithmetics/ecdsa/ecdsa_intro.tex`

### Configuration

| Parameter | Value |
|-----------|-------|
| Curve | secp256k1 |
| Method | Shamir's trick double-and-add |
| N_rows | 258 |
| Witness columns | 14 |
| Prime tuple | three components |

### Three Trace Components

| Component | Ring | Purpose |
|-----------|------|---------|
| Q[X]-trace | Q[X] | bit columns |
| F_p[X]-trace | F_p[X] | curve arithmetic |
| F_n[X]-trace | F_n[X] | scalar accumulation |

### Constraints

- 9 F_p constraints + 2 F_n constraints per row
- Maximum constraint degree: 6

---

## 18. Implementation Mapping

### Prover Pipeline Order (from AirZinc)

```
1. Encode witness → commit (Merkle)          [zip-plus/src/pcs/]
2. Receive prime q_0, OOD points              [snark/src/pipeline.rs]
3. Send OOD evaluations                       [snark/src/pipeline.rs]
4. Ideal Check (MLE eval + membership)        [piop/src/ideal_check/]
5. Evaluation Projection                      [piop/src/projections.rs]
6. Constraint batching (CPR)                  [piop/src/combined_poly_resolver/]
7. Sumcheck                                   [piop/src/sumcheck/]
8. Lift-and-project for oracle evals          [piop/src/projections.rs]
9. Batched shift protocol                     [piop/src/shift_sumcheck/]
10. LogUp lookup                              [piop/src/lookup/]
11. Batched IOPP (Zip+)                       [zip-plus/src/pcs/]
```

### Verifier Pipeline Order

```
1. Receive commitments
2. Send prime q_0, OOD points, prime m
3. Receive OOD evaluations
4. Check ideal membership of MLE evaluations
5. Send projection points a_i
6. Receive constraint batching results
7. Run sumcheck verifier rounds
8. Verify lift-and-project consistency
9. Verify shift reduction
10. Verify LogUp
11. Run IOPP verifier (spot checks)
12. Accept/reject
```

### Crate Mapping

| Paper Component | Crate | Module |
|----------------|-------|--------|
| UCS / UAIR+ | `uair/` | Relation definitions |
| Zinc+ PIOR Compiler | `snark/src/pipeline.rs` | End-to-end pipeline |
| Ideal Check | `piop/src/ideal_check/` | `ideal_check.rs` + submodules |
| CPR | `piop/src/combined_poly_resolver/` | `combined_poly_resolver.rs` + submodules |
| Sumcheck | `piop/src/sumcheck/` | `sumcheck.rs` + submodules |
| Projections | `piop/src/projections.rs` | φ_q and ψ_{q,a} maps |
| Shift protocol | `piop/src/shift_sumcheck/` | `shift_sumcheck.rs` + submodules |
| LogUp lookup | `piop/src/lookup/` | Lookup submodules |
| Random field sumcheck | `piop/src/random_field_sumcheck.rs` | Supplementary sumcheck |
| Zip+ IOPP/PCS | `zip-plus/src/pcs/` | Commit/prove/verify |
| IPRS encoding | `zip-plus/src/code/` | Code construction |
| Merkle tree | `zip-plus/src/merkle/` | Hash-based commitment |
| MLE polynomials | `poly/src/mle/` | Multilinear extensions |
| Univariate polys | `poly/src/univariate/` | Univariate operations |
| SHA-256 UAIR+ | `sha256-uair/` | Constraint + witness gen |
| ECDSA UAIR+ | `ecdsa-uair/` | Constraint + witness gen |
| Transcript | `transcript/` | Fiat-Shamir transcript |

### Key Formula Cross-Reference

| Paper Formula | Where to Check in Code |
|--------------|----------------------|
| `ṽ(r) = Σ_b Q(b,y,f)·eq(b;r)` | `piop/src/ideal_check/` |
| `Σ_j α_j·Q'_j(r,...) · eq(r,ρ) = 0` | `piop/src/combined_poly_resolver/` + `piop/src/sumcheck/` |
| `shiftdown(r,b) · ṽ(b)` sumcheck | `piop/src/shift_sumcheck/` |
| `Σ 1/(β-w_i) = Σ m_j/(β-t_j)` | `piop/src/lookup/` |
| `g_i(0) + g_i(1) = s_{i-1}` | `piop/src/sumcheck/` |
| `w·u*_t = α*_t` (inner product) | `zip-plus/src/pcs/` |
| `‖Enc(x)‖ ≤ ‖x‖·(q/2)^{d+1}·k` | `zip-plus/src/code/` |
| `φ₂(â)·ρ₀(X) - φ₂(ŷ) ∈ (X^w-1)` | `sha256-uair/src/constraints.rs` |
| `ŷ - Σx_j + μ·X^w ∈ (X-2)` | `sha256-uair/src/constraints.rs` |

---

## Appendix: Key Soundness Parameters

### Zinc+ PIOR Soundness

- Prime projection: depends on `|primeset|` and degree of constraints
- Ideal check: `μ / |S|` per constraint (via Schwartz-Zippel)
- Evaluation projection: standard Schwartz-Zippel over extension field

### Zip+ IOPP Soundness

- Sumcheck: degree/|F| per round
- Proximity: `(1 - β)^C` for C spot checks
- List-decoding: `(J-1)·err_pg + L/2^{K-1}` for proximity gap

### Compiled IOP Soundness

- Out-of-domain sampling: `s_1·(2L²/|P| + L²·μ/|R|) + Σ L²_qs·s_qs·μ_qs/|R_qs|`
- Poly-IOP rounds: `ε^poly_i` per round
- IOPP rounds: `Σ_qs ε^prox,qs_i` per round
- Total round complexity: `1 + R_poly + R_prox`

### Fiat–Shamir

The compiled IOP satisfies RBR knowledge soundness, which enables Fiat–Shamir compilation to a non-interactive argument of knowledge in the ROM (via [BCFW25]).
