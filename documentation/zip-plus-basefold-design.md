# Zip+ basefold variant — design ledger

Branch: `zip-plus-basefold` (off `main-beta`). Status: milestones 1-2 done
(interactive IOPP + Merkle/Fiat-Shamir compilation with spot-check queries).

This branch implements the **limbwise-folding (BaseFold-style) variant of Zip+**:
a hash-based IOPP over the integers with polylogarithmic verification, following
the draft paper *"Zip++: Hash-Based IOPPs over the Integers with Polylogarithmic
Verification via Limbwise Folding"* (working draft in
`~/zinc- whir idea/zip-plus-plus-draft/`, Albert, June 2026). Read that draft
first; this file maps it onto the codebase and records implementation decisions.

## Protocol recap (one paragraph)

FRI-style folding over Z fails because each fold adds ~lambda bits to entries.
Fix: keep the round-r state as k_r *limb vectors* with balanced base-2^p digits
(virtual word = sum_j 2^{p(j-1)} v_j); each round splits every limb into
even/odd halves, folds **all** halves with fresh independent lambda-bit
challenges (the radix weights never enter the fold), and re-decomposes the
folded vector into fresh limbs. For p >= lambda + 2 the limb count stabilizes
at k = 2 (arity 2; p >= lambda + s + 1 at arity 2^s). Evaluation claims (which
Zinc+ needs *projected*: values in F_{q0} for the transcript-sampled prime q0)
are tracked **linearly**: per-round half-claims checked with weights
(1 - z_r, z_r), folded claim = alpha-combination, decomposition reflected by
sum_j 2^{p(j-1)} e_j = e'; final limbs sent in clear and checked directly. No
sumcheck. Consistency across rounds is checked at spot positions via the
**lifted butterfly**: c_j = A + tau_j B, c_{j+n/2} = A - tau_j B with
tau_j = centered_lift(omega^j), so the verifier derives tail-code symbols from
two openings. The codes must form a full-depth radix-2 IPRS *chain* (over Z the
FFT depth is part of the code; fold rounds <= depth; the unfolded tail is sent
in clear).

## Module layout

```
zip-plus/src/basefold/
  mod.rs       module doc + error type
  limbs.rs     balanced base-2^p decomposition / recombination over Int<N>
  chain.rs     radix-2 foldable IPRS chain (encode at every level; twiddles)
  iopp.rs      interactive prover + verifier (full oracles, all positions);
               also hosts the shared round/claim/consistency helpers
  compiled.rs  Merkle-compiled prover + verifier: pair-bundled leaves
               (leaf j of round r = (u_t[j], u_t[j+n_{r+1}]) for all limbs t,
               one tree per round), roots absorbed instead of oracles,
               Q spot checks at chained positions j_{r-1} = s mod n_r with
               one leaf opening per round per query; CompiledProof +
               size_bytes accounting
```

`pub mod basefold;` added to `zip-plus/src/lib.rs` — the only touch to
existing files in milestone 1.

## Type decisions

- Reuse `crypto_primitives` fixed-width ints, mirroring `pcs/test_utils.rs`:
  digits/witness `Int<ND>`, codewords `Int<NK>`, consistency-check accumulators
  `Int<NW>` (wide), challenges `Int<NC>` sampled as CHAL_BITS-bit non-negative
  ints. Test instantiation: ND=3 (192b, holds p=130-bit digits), NK=8 (512b),
  NW=8, CHAL_BITS=128.
- Claims live in a generic `F: PrimeField` with runtime config (`MontyField`),
  sampled via `transcript.get_random_field_cfg::<F, Uint<K>, MillerRabin>()` —
  same mechanism as the existing PCS (q0 = random ~256-bit prime in tests).
- Chain twiddles: i64 centered lifts of F65537 roots of unity (same
  `normalize` convention as `code/iprs/pntt`); base case = lifted Vandermonde.
  n | q - 1 caps codeword length at 2^16 with F65537 — fine for tests; larger
  base fields are a later config knob.
- **No rationals anywhere.** The verifier's butterfly inversion
  (A = (c1+c2)/2, B = (c1-c2)/(2 tau)) is never materialized; every
  consistency check is multiplied through by 2*tau_j and verified as an
  integer identity:
    2 tau_j * RHS_recombined  ==  sum_t [ alpha_e_t * tau_j * (c1_t + c2_t)
                                        + alpha_o_t * (c1_t - c2_t) ].
- MLE convention: `DenseMultilinearExtension` pairs index parity with
  `point[0]` (checked in `poly/src/mle/dense.rs::fix_variables`), so the
  even/odd split consumes z coordinates front-to-back — matches the draft.

## Conventions honored

- Clippy: no raw arithmetic in lib paths — `add!`/`mul!`/`sub!` macros or
  scoped, justified `#[allow(clippy::arithmetic_side_effects)]`; no `unwrap`
  in lib code (tests carry a module-level allow like `pcs/test_utils.rs`).
- F_2-ledger rule: this work is on the integer side and adds new modules only;
  if any shared in-scope file (merkle.rs, pcs/phase_commit.rs, ...) gets
  touched, add the `documentation/f2x-sha-todo.md` entry per CLAUDE.md.

## Milestones

1. **done** `limbs.rs` + `chain.rs` + interactive `iopp.rs` (explicit
   oracles, challenges from a transcript, consistency checked at all
   positions), completeness + tamper tests.
2. **done** Merkle per round (leaf = pair-bundled positions {j, j + n_{r+1}},
   all limbs in one leaf) + Fiat–Shamir query phase (j_{r-1} = s mod n_r
   chaining), CompiledProof + size accounting. Toy-scale datapoint:
   ~84 KB at m = 2^8, R = 6, Q = 30 (query openings dominate, as expected;
   real sizing needs m >= 2^16 and the parameter work of milestone 3).
3. Parameters: arity-8 folds (reuse radix-8 butterflies), bigger base field
   (5*2^25+1) for long rows, depth dial, measured gamma at depth 4-6.
4. Integration: `LinearCode`/`ZipTypes`-shaped adapter, batching
   (batch-then-fold round 0 on the interleaved commitment), wiring into
   `protocol` as an alternative opening for the int lane; benches.

## Open design questions (tracked, not blocking)

- Negative-entry witnesses: balanced digits handle signs; round-0 witness with
  k_0 > 1 limbs deferred (tests use entries < 2^{p-1}, k_0 = 1).
- Soundness-slack bound B* = (p+2)R + p k_0 must be threaded into the outer
  protocol's bitsize accounting when integrating (draft Thm 6.1).
- Query count / proximity regime: unique-decoding first; 1.5-Johnson later.
