# Batched Decomposition + LogUp Protocol

## Overview

The batched Decomposition + LogUp protocol proves that **L witness vectors**
all have entries belonging to the same decomposed lookup table, using a
**single sumcheck** instance. This amortises the dominant prover cost across
all lookups.

For a table of size $2^{K \cdot c}$ (e.g. `BitPoly(32)`), the table is
decomposed into $K$ sub-tables of size $2^c$. Each witness $w^{(\ell)}$ is
split into $K$ chunks:

$$
w^{(\ell)}_i = \sum_{k=0}^{K-1} \text{shifts}[k] \cdot c^{(\ell)}_k[i]
$$

where each chunk $c^{(\ell)}_k[i]$ must lie in the shared sub-table $T$.

## Protocol Steps

### Prover

1. **Send chunks** in the clear: all $L \cdot K$ chunk vectors
   $c^{(\ell)}_k$.

2. **Compute multiplicities** per chunk using a shared table index, then
   aggregate per lookup:
   $$
   m_{\text{agg}}^{(\ell)}[j] = \sum_{k=0}^{K-1} m_k^{(\ell)}[j]
   $$
   where $m_k^{(\ell)}[j]$ counts occurrences of $T[j]$ in chunk $k$ of
   lookup $\ell$. Send all $L$ aggregated multiplicity vectors.

3. **Derive challenge** $\beta$ from the transcript.

4. **Compute inverse vectors**:
   - Per-chunk: $u^{(\ell)}_k[i] = 1/(\beta - c^{(\ell)}_k[i])$
   - Shared table: $v[j] = 1/(\beta - T[j])$ (computed once)

   Send all inverse vectors.

5. **Derive batching challenge** $\gamma$.

6. **Precompute** the batched identity polynomial $H$ pointwise:
   $$
   H[j] = \sum_{\ell=0}^{L-1} \left[
     \sum_{k=0}^{K-1} \gamma^{b_\ell + k} \bigl((\beta - c^{(\ell)}_k[j]) \cdot u^{(\ell)}_k[j] - 1\bigr)
     + \gamma^{b_\ell + K} \bigl(\sum_k u^{(\ell)}_k[j] - m_{\text{agg}}^{(\ell)}[j] \cdot v[j]\bigr)
   \right]
   $$
   where $b_\ell = \ell \cdot (K+1)$.

7. **Run sumcheck** on $\sum_x \text{eq}(x, r) \cdot H(x) = 0$ with only
   **2 MLEs** at **degree 2**.

### Verifier

1. **Mirror transcript** operations (absorb chunks, multiplicities,
   inverses) and derive the same $\beta$, $\gamma$, $r$ challenges.

2. **Verify sumcheck** proof.

3. **Direct table-inverse check**: confirm $(\beta - T[j]) \cdot v[j] = 1$
   for all $j$.

4. **Recompute** $H(x^*)$ at the sumcheck's claimed point $x^*$ via MLE
   inner products with the precomputed $\text{eq}(\cdot, x^*)$ vector:
   $$
   H(x^*) = \sum_{\ell,k} \gamma^{b_\ell+k}\bigl(\beta \cdot \tilde{u}_k(x^*) - \widetilde{c_k \cdot u_k}(x^*) - 1\bigr)
   + \sum_\ell \gamma^{b_\ell+K}\bigl(\sum_k \tilde{u}_k(x^*) - \widetilde{m_{\text{agg}} \cdot v}(x^*)\bigr)
   $$

5. **Check** $\text{eq}(x^*, r) \cdot H(x^*) = \text{subclaim value}$.

6. **Multiplicity sum check**: $\sum_j m_{\text{agg}}^{(\ell)}[j] = K \cdot W$
   for each lookup $\ell$ (where $W$ is the witness length).

## Identities

For each lookup $\ell$ (with offset $b_\ell = \ell \cdot (K+1)$):

| Index | Identity | Purpose |
|-------|----------|---------|
| $\gamma^{b_\ell + k}$ for $k = 0 \ldots K-1$ | $(\beta - c^{(\ell)}_k) \cdot u^{(\ell)}_k - 1 = 0$ | Inverse correctness |
| $\gamma^{b_\ell + K}$ | $\sum_k u^{(\ell)}_k - m_{\text{agg}}^{(\ell)} \cdot v = 0$ | Log-derivative balance |

**Total**: $L \cdot (K+1)$ identities, all $\gamma$-batched into one polynomial.

## Key Optimization: Degree Reduction

The non-batched `DecompLogupProtocol` passes $2K+3$ MLEs to the sumcheck
with degree 3 (the combination function multiplies `eq` by products of two
data MLEs). The batched version instead **precomputes** $H$ pointwise before
the sumcheck:

| | Non-batched | Batched |
|---|---|---|
| MLEs in sumcheck | $2K + 3$ | **2** (`eq`, `H`) |
| Sumcheck degree | 3 | **2** |
| Sumcheck rounds | same `num_vars` | same `num_vars` |

This is valid because each identity is a fixed arithmetic expression over the
auxiliary vectors at each evaluation point — the $\gamma$-weighted sum can be
collapsed into a single MLE before entering the sumcheck. The product
`eq · H` is degree 2 (product of two multilinear polynomials).

## Decomposition Consistency

The protocol does **not** verify decomposition consistency internally
($w^{(\ell)}_i = \sum_k \text{shifts}[k] \cdot c^{(\ell)}_k[i]$). This
check is delegated to the outer protocol that invokes the batched lookup.

## Data Structures

- **`BatchedDecompLookupInstance`**: Input with L witnesses, shared
  sub-table, shifts, and per-witness chunks.
- **`BatchedDecompLogupProof`**: Contains chunk vectors, aggregated
  multiplicities, inverse witnesses, shared inverse table, and the single
  sumcheck proof.
- **`BatchedDecompLogupVerifierSubClaim`**: The evaluation point and
  expected value returned to the outer protocol.
