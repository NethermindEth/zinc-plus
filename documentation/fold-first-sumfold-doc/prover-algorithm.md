# Fold-First SumFold Prover Algorithm (Gram-Form Univariate Skip)

This note is the implementation blueprint for the V2 fresh-batch prover: SumFold
runs **before** IdealCheck and scalarization, the SumFold sumcheck is a
booleanity **zerocheck** over the instance axis realized as a single
univariate-skip round, and the skip round is computed through the Gram-matrix
sufficient statistic so the prover pays **no per-interpolation-point work**.

Protocol-level soundness (the three-lock argument for why IdealCheck is still
respected post-fold) lives in the companion spec note
(`NeutronNova + Zinc+ Fold-First SumFold Protocol`); this file is prover-side
only: the step-by-step algorithm, data structures, cost model, and the full
optimization catalog.

References: univariate skip over large prime fields — Dao–DeStefano–Bagad–Domb–
Thaler, *Speeding Up Sum-Check Proving*, §7; Gruen, *Some Improvements for the
PIOP for ZeroCheck* (ePrint 2024/108); NeutronNova (ePrint 2024/1606).

---

## 0. Setting

$N$ fresh SHA instances, $j \in \{0,\dots,N-1\}$ ($N$ need **not** be a power of
two). Row domain $H_{\mathrm{row}} = \{0,1\}^7$ (128 rows, live rows $0..71$).
Booleanity source catalog $\mathcal{B}$ (real word-bit slices + virtual Ch/Maj),
residual families $R_0,\dots,R_{17}$, all affine in $(w, y)$;
$\mathcal{F}_{\neq 0} = \{0,1,4,5,6,9,10\}$.

Skip domain and basis (statement-bound, verifier-derived):

$$
D_N = \{u_0,\dots,u_{N-1}\} \subset K \text{ (symmetric integers around } 0),
\qquad
L_j(Y) = \prod_{i\neq j}\frac{Y-u_i}{u_j-u_i},
\qquad
Z_{D_N}(Y) = \prod_j (Y-u_j).
$$

Partition of unity $\sum_j L_j(Y) \equiv 1$ is what lets affine residual parts
and shared public columns fold consistently.

---

## 1. Prover algorithm, step by step

### P0 — Prepare (witness-dependent, challenge-independent; cacheable)

For every instance $j$, synthesize the projected trace (bit slices
$B_{j,m,k}[z] \in \{0,1\}$, int columns, publics) as today, **plus** build the
transposed booleanity view:

- For each booleanity source $q \in \mathcal{B}$ and row $z$: one **instance-major
  packed word** $\mathsf{mask}[q][z] \in \{0,1\}^N$ (a `u32`/`u64`/`u128` or
  small bit array), where bit $j$ is $D_{j,q}(z)$.
- Real sources: a transpose of the existing per-instance bit slices.
- Virtual Ch/Maj sources: **not materialized** — store the recipe; masks are
  computed on the fly in P2 via bitwise ops on real masks (see O-10).

This replaces the per-instance residual bases inside
`prepare_sha_sumfold_basis` for SumFold purposes; the bit slices themselves are
still needed for commitment and for the fold in P4.

Cost: one pass over all bits, $|\mathcal{B}|\cdot 128 \cdot N$ bit moves.
Lives in the existing `prepare` phase (outside the online prove path).

### P1 — Commit, absorb, first challenges

$$
C_j = \operatorname{Com}(w_j), \qquad
\mathsf{tr} \leftarrow \mathsf{absorb}(\{C_j\},\{y_j\}), \qquad
r_{\mathrm{ic}} \leftarrow K^7, \quad \rho \leftarrow K.
$$

Unchanged commitment path, plus one instance-batching challenge
$\gamma \leftarrow K$ (power weights $\gamma^j$ — the $m$-ary replacement for
the $\operatorname{eq}(\beta,\cdot)$ tensor; works for any $N$, not just powers
of two). Not sampled here: $a, \lambda, \xi$ (post-fold).

### P2 — Gram pass (the SumFold accumulator)

Weights, built once:

$$
\omega_{q,z} = \rho^{\operatorname{idx}(q)} \cdot \operatorname{eq}(r_{\mathrm{ic}}, z)
\qquad
(|\mathcal{B}| \cdot 128 \text{ sb mults from cached } \rho\text{-powers} \times \text{eq-row table}).
$$

Sufficient statistic — one streaming pass over $(q, z)$:

$$
G[j,k] \mathrel{+}= \omega_{q,z} \quad \forall\, j \le k \in S_{q,z},
\qquad
S_{q,z} := \{ j : \mathsf{mask}[q][z]_j = 1 \},
$$

with $G$ an upper-triangular $N \times N$ array of **unreduced wide
accumulators**. For the honest prover $h[j] = G[j,j]$ (bits: $d_j^2 = d_j$), so
no separate $h$ pass is needed (debug-recompute it, see O-16).

```rust
// hot loop sketch — no multiplications, adds only
for (q, z) in booleanity_items {          // parallel-sharded, see O-12
    let mask = masks.get(q, z);           // virtual: computed on the fly
    if mask == 0 { continue; }            // padding rows / inactive: free
    let omega = &omega_tab[q][z];         // 4-limb reduced field element
    let mut m1 = mask;
    while m1 != 0 {
        let j = m1.trailing_zeros();
        let mut m2 = m1;                  // k >= j  (upper triangle)
        while m2 != 0 {
            let k = m2.trailing_zeros();
            gram[tri(j, k)].add_unreduced(omega);   // 512-bit accumulator
            m2 &= m2 - 1;
        }
        m1 &= m1 - 1;
    }
}
// one Barrett per entry afterwards: N(N+1)/2 reductions total
```

The polynomial identity this computes (with $L(Y) = (L_0,\dots,L_{N-1})^\top$):

$$
q(Y) = L(Y)^\top G\, L(Y) - h^\top L(Y),
\qquad
\deg q \le 2(N-1),
$$

and $q(u_j) = G[j,j] - h[j] = 0$ for honest bits — the zerocheck holds by
construction.

Cost: $\binom{|S|+1}{2}$ wide adds per item, $\mathbb{E}|S| \approx N/2
\Rightarrow \approx N^2/8$ adds per item, **independent of the number of
evaluation points**.

### P3 — Finish: skip-round message, $\alpha$, fold weights

Textbook univariate-skip transmission (no quotient — matches BDDT §7 and the
standard skip literature):

1. Statement-fixed node set: the $N$ domain points $\{u_j\}$ plus $N-1$
   off-domain integers $\{y_s\}$ — $2N-1$ nodes, enough for
   $\deg q \le 2(N-1)$.
2. Evaluate the quadratic form at the **off-domain nodes only**:
   $q(y_s) = L(y_s)^\top G L(y_s) - h^\top L(y_s)$ — $O(N^2)$ mults per point,
   $O(N^3)$ total (sb on the symmetric integer domain). The honest domain
   values are free: $q(u_j) = G[j,j] - h[j] = 0$.
3. Transmit $q$ in **evaluation basis** over the node set ($2N-1$ values; the
   $N$ domain values are $0$ for an honest prover).
4. $\mathsf{tr} \leftarrow \mathsf{absorb}(q)$. Verifier: degree bound is
   enforced by the fixed node count; check the batched zerocheck equation

$$
\sum_{j=0}^{N-1} \gamma^j\, q(u_j) \overset{?}{=} 0
\qquad (O(N) \text{ — domain values are read directly}),
$$

   then sample $\alpha \leftarrow K$ and set

$$
\theta_j = L_j(\alpha) \ \ (\text{barycentric, one inversion}),
\qquad
B^\star = q(\alpha) \ \ (\text{barycentric over the } 2N-1 \text{ nodes}).
$$

The $\gamma$-weights are **load-bearing**: with an unweighted sum a cheater
cancels nonzero per-instance residues across instances ($q_1 = c$,
$q_2 = -c$), and $B^\star = q(\alpha)$ then folds the invalid instance in
undetected. Random weights kill the cancellation w.h.p.; the $\alpha$-endpoint
plus the row-sumcheck/opening chain kills any transmitted $\widetilde q \neq q$
w.h.p.

*Optional compression (non-standard):* pin the domain values to zero
structurally and send only the $N-1$ off-domain values — equivalently the
quotient $q/Z_{D_N}$. Saves $N$ field elements and upgrades per-instance
pinning from w.h.p. to exact, at the cost of a message format not found in the
skip literature. Default to the weighted form above.

### P4 — Fold

$$
w' = \sum_j \theta_j w_j, \qquad
y' = \sum_j \theta_j y_j, \qquad
C' = \sum_j \theta_j C_j .
$$

One pass over all witness columns (bit slices fold to field-valued columns).
This is `fold_projected_traces` — a cost common to V1 and V2.

### P5 — Folded IdealCheck (once)

From the **folded** columns only:

$$
E'_f(X) = \sum_z \operatorname{eq}(r_{\mathrm{ic}}, z)\, C_f(z, X;\, w', y'),
\qquad f \in \mathcal{F}_{\neq 0}.
$$

Transmit 7 coefficient vectors; verifier checks degree bounds + membership in
$(X^{32}-1)$ / $(X-2)$, absorbs. Then sample $a, \lambda, \xi$. This replaces
the per-instance ideal cache + $\beta$-aggregation with one instance's worth of
work.

### P6 — Scalarize once, post-fold

$$
W_m'^{(a)}[z] = \sum_{k=0}^{31} a^k B'_{m,k}[z]
\qquad (11 \cdot 128 \cdot 32 \text{ bb mults} \approx 45\text{k — done once, not } N\text{ times}).
$$

Direct scalarization of the folded trace beats folding $N$ pre-scalarized
tables; there is exactly one scalarization event in the whole protocol.

### P7 — Folded row sumcheck (unchanged mechanics)

Assembled target and claim:

$$
T' = \sum_{f\in\mathcal{F}_{\neq 0}} \lambda^f E'_f(a) + \xi B^\star
\ \overset{?}{=}\
\sum_z \operatorname{eq}(r_{\mathrm{ic}}, z)\big[L'(z) + \xi B'(z)\big],
$$

with $L', B'$ built from the folded trace ($D'_q(z) = \sum_j \theta_j
D_{j,q}(z)$ read off the folded bit slices). Existing 7-round row-sumcheck
prover. Debug cross-check: $q(\alpha) = \sum_z \operatorname{eq}(r_{\mathrm{ic}},z) B'(z)$.

### P8 — Endpoints, multipoint, lifted evals, PCS opening

Byte-for-byte the existing tail: $r^\star$ endpoint claims, scalarization links
$W_m'^{(a)}(r^\star) = \sum_k a^k B'_{m,k}(r^\star)$, shifted-source links,
virtual reconstruction, multipoint to $r_0$, lifted evals, Hyrax against $C'$.

---

## 2. Cost model

Per-item Gram adds $\approx N^2/8$; items $= |\mathcal{B}| \cdot |\text{live rows}|$.
Measured V1 baselines from the 2026-07-02 phase probe (HEAD `9f8b814`, N=32,
$\ell_0=3$): prepare_sumfold_basis 36.3 ms, aggregate_ideal 5.8 ms, sumfold
linear 1.35 ms, booleanity accumulator 5.85 ms, sumfold rounds 6.26 ms.

| Step | Scaling | N=8 | N=32 | N=128 |
|---|---|---|---|---|
| P2 Gram pass (wide adds) | items · N²/8 | ~0.7M | ~11M | ~165M |
| P3 finish (sb mults) | ~N³/2 | 256 | 16k | 1M |
| P5 folded IdealCheck | 1 instance | — | ~0.2 ms | ~0.2 ms |
| P6 scalarization | 1 instance | ~45k bb | ~45k bb | ~45k bb |
| V1 work replaced | per-instance | ~10 ms | ~50 ms | ~200 ms (extrap.) |

The Gram pass parallelizes embarrassingly ($G$ copy per thread = $N^2/2$
accumulators ≈ 32 KB at N=32, L1-resident; merge once). Expected online SumFold
block at N=32: ~1–2 ms vs ~12 ms today, plus deletion of most of the 36 ms
prepare-basis and 5.8 ms aggregate-ideal phases.

Proof size: $-4\ell_b$ round evaluations, $+(2N-1)$ skip-round evaluations
($N$ of them zero for an honest prover; the quotient variant compresses to
$N-1$), ideal polys unchanged (7 either way).

---

## 3. Optimization catalog

### Algorithmic

- **O-1 Gram sufficient statistic.** $(G, h)$ makes the hot pass independent of
  the number of interpolation points; all evaluations become $O(N^2)$ quadratic
  forms afterwards. This is the headline: the "interpolation terms" cost is
  removed, not reduced.
- **O-2 Upper-triangle symmetry.** $G$ symmetric ⇒ accumulate $j \le k$ only;
  halves adds and memory. Diagonal handled in the same set-bit loop.
- **O-3 Weighted zerocheck round (default); quotient as optional compression.**
  The skip round itself is textbook: transmit $q$ (evaluation basis, $2N-1$
  nodes) and check $\sum_j \gamma^j q(u_j) = 0$ — no quotient, matching the
  univariate-skip literature. The weights are soundness-load-bearing (an
  unweighted sum admits cross-instance cancellation). Optional non-standard
  variant: send $q/Z_{D_N}$ ($N-1$ values), which pins the $N$ zeros exactly
  and drops $\gamma$ — adopt only if skip-message size at large $N$ ever
  matters.
- **O-4 $h$ from the diagonal.** Honest bits give $h[j] = G[j,j]$; skip the
  separate $h$ pass. Keep a debug-mode recomputation (see O-16).
- **O-5 Empty-support skip.** Padding rows and selector-inactive sources have
  all-zero masks and cost zero — only live rows and active sources are touched.
  No explicit active-row bookkeeping needed; the `mask == 0` test does it.
- **O-6 Batched inversion.** The barycentric denominators for
  $\theta_j = L_j(\alpha)$ and for evaluating $B^\star = q(\alpha)$ from the
  $2N-1$ node values via one Montgomery batch inversion each.
- **O-7 Symmetric integer domain for the finish.** $D_N$ symmetric around 0 ⇒
  $L_j(y_s)$ lift to small integers ⇒ the $O(N^3)$ finish is sb, not bb. (With
  the Gram form this is a convenience, not a constraint — the paper's
  $B\cdot\Lambda < 2^{127}$ window bound applied to the *direct* subset-sum
  path, which no longer exists. Full-width packing is fine at any practical $N$.)
- **O-8 Large-$N$ variants.** If the $N^2/8$-per-item pass ever dominates
  (N ≥ 128): (a) block-bucketed Gram — per item add $\omega$ into
  $\lceil N/b \rceil^2$ block-pair buckets indexed by (mask$_u$, mask$_v$),
  expand once at the end ($2^{2b}$ buckets per block pair — memory/compute
  trade at $b \in \{4,\dots,6\}$); (b) two-stage folding (fold 32 at a time)
  once relaxed-accumulator lanes exist.
- **O-9 $N$ arbitrary.** $m$-ary packing does not need power-of-two $N$; drop
  the `InstanceCountNotPowerOfTwo` restriction for this path.

### Data layout

- **O-10 Instance-major bit-packing.** One $N$-bit word per $(q, z)$
  (`u32`/`u64`/`u128`). Built once in P0 by transposing the existing bit-slice
  MLEs. Virtual Ch/Maj masks are *never materialized as columns*: compute
  `mask_ch1 = mask_E_shift2 ^ mask_E_shift1 ...`-style bitwise recipes on the
  fly from real-source masks (word-parallel across all $N$ instances at once).
  Note: virtual sources are affine combinations that are 0/1-valued for honest
  traces, so their masks are exact bit masks; the recipe mirrors
  `reconstruct_virtual_ch_maj_at_row`.
- **O-11 Fused weight table.** Materialize $\omega_{q,z}$ once as reduced field
  elements via the outer product of cached $\rho$-powers and the eq-row table
  (sb mults); the Gram loop then reads one pointer per item and does adds only.
- **O-12 Sharded parallelism with L1-resident state.** Shard items over threads
  (by source block or row block); each thread owns a private $G$ copy
  ($N(N+1)/2$ wide accumulators ≈ 32 KB at N=32); merge once at the end
  ($N^2/2$ wide adds per thread — trivial). No atomics, no false sharing.
- **O-13 Streaming order.** Iterate $z$ outer, $q$ inner (or vice versa) so the
  $\omega$ table and mask arrays stream sequentially; the only random access is
  into the L1-resident $G$.

### Arithmetic

- **O-14 Delayed reduction everywhere.** $G$ entries are 512-bit (8-limb)
  unreduced accumulators; the hot loop is pure limb adds with carries — no
  multiplications, no reductions. One Barrett per entry at the end
  ($N(N+1)/2 \le 528$ reductions at N=32). Overflow headroom: each add ≤ 4-limb
  value; $2^{256}$ headroom ⇒ safe for $2^{256}$ adds — no mid-pass flush ever.
- **O-15 Set-bit iteration.** `trailing_zeros` + `m &= m-1` loops (as sketched)
  beat per-bit branches; for dense masks consider iterating bytes with a
  256-entry offset table. SIMD (NEON/AVX2) for the 8-limb accumulator adds —
  2–4 vector ops per add.
- **O-16 Debug-only integrity checks.** (a) Recompute $h$ independently and
  assert $h[j] = G[j,j]$ (catches transpose/mask bugs = catches non-boolean
  witness bits); (b) assert $q(u_j) = 0$ via the quadratic form at domain
  points; (c) assert $q(\alpha) = \sum_z \operatorname{eq}(r_{\mathrm{ic}},z) B'(z)$
  after the fold (ties skip round to row-sumcheck inputs). All behind
  `debug_assertions`.
- **O-17 Precomputed integer Lagrange tables.** For the fixed statement domain
  $D_N$: integer values $L_j(y_s)$ and products $L_j(y_s)L_k(y_s)$ (or evaluate
  the quadratic form as $v = G \cdot L(y_s)$ then $L(y_s)^\top v$ — $N^2 + N$
  mults, no product table). Barycentric weights for $\theta$ likewise
  statement-static.

### Pipeline-level (what V2 deletes or shrinks — do not keep these alive)

- **O-18 Delete per-instance ideal caches** (`build_fresh_sha_ideal_cache`,
  $\beta$-aggregation — the 5.8 ms `aggregate_ideal` phase at N=32). Replaced
  by P5 on the folded trace.
- **O-19 Delete per-instance scalarization and the linear accumulator**
  ($Q_L$, $Q_0$, `build_sha_sumfold_linear_accumulator*`, per-instance
  $W_{i,m}^{(a)}$ tables and the residual bases feeding them — the bulk of the
  36 ms `prepare_sumfold_basis` phase). The linear part folds exactly; nothing
  to sumcheck.
- **O-20 Delete the ternary booleanity machinery** ($\{\infty,0,1\}^{\ell_0}$
  grids, `BoolQuadExtrapolate`, prefix/tail hybrid state, tail-round binding —
  the 5.85 + 6.26 ms phases). Replaced by P2 + P3.
- **O-21 Scalarize-once ordering.** P6 after the fold: 45k bb mults total
  instead of $N \times$ subset-sum scalarization + $N$-way table folds.
- **O-22 Row-sumcheck input reuse.** $B'(z)$ and $W'^{(a)}$ come from the same
  folded trace pass; fold bit slices column-blocked so P4 output feeds P5/P6/P7
  without re-walking memory.
- **O-23 Commitment fold as one MSM** (verifier side, already noted in the perf
  baseline): $C' = \sum_j \theta_j C_j$ deferred into the final opening MSM.

### Transcript / proof format

- **O-24 New round-message variant.** `MultiDegreeSumcheckProof` needs a
  skip-round message type (evaluation vector of $q$ over the $2N-1$
  statement-fixed nodes) alongside the existing per-round evaluations; verifier
  checks $\sum_j \gamma^j q(u_j) = 0$, samples one $\alpha$, computes
  $B^\star = q(\alpha)$ and $\theta = L(\alpha)$ barycentrically ($O(N)$, one
  batch inversion). The `PrefixFastPath` / `Round1FastPath` hooks in
  `piop/src/sumcheck/multi_degree.rs` are the natural seam, but the message
  shape and verifier change make this a proof-format break — new domain label
  `PF_CONCISE_SHA256_FRESH_BATCH_V2`.
- **O-25 Accumulator lane (future).** A relaxed lane with target
  $T_{\mathrm{acc}} \neq 0$ is trivial in the weighted form: the check becomes
  $\sum_j \gamma^j q(u_j) = \gamma^{j_{\mathrm{acc}}} T_{\mathrm{acc}}$; the
  lane's (field-valued) row enters $G$ via ordinary sb mult-adds instead of
  masked adds — one dense lane among $N$.

---

## 4. Code touchpoints

| Area | File | Change |
|---|---|---|
| Transposed masks + recipes | `piop/src/neutron_nova/booleanity.rs` (or new `gram.rs`) | new: P0 transpose, P2 Gram pass |
| Skip round + quotient | `piop/src/sumcheck/multi_degree.rs` | new message variant, verifier round |
| SumFold entry | `piop/src/neutron_nova/sumfold.rs`, `projection_sha.rs` | replace group builders for V2 path |
| Pipeline order | `protocol/src/production_sha.rs` | new phase order (P1–P8), new labels |
| Deleted paths | `projection_sha.rs` linear/ideal-cache/ternary builders | keep behind V1 feature for differential tests |
| Bench | `protocol/benches/e2e.rs`, `piop/benches/neutron_nova_sumfold.rs` | V1-vs-V2 same-binary A/B, phase probes |

Validation plan: differential test V1 vs V2 on identical traces (folded claim
and accepted/rejected agreement), per-family tamper tests (flip one bit slice /
one residual — each must flip the verdict), the O-16 asserts, and the standing
rule from the perf baseline: same feature set (at least `parallel`),
same-binary A/B only.
