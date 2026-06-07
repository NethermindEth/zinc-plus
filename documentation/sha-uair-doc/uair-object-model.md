# UAIR Object Model for Production SHA-256

This note records the naming and lifecycle decisions for the generic UAIR
objects we want to use before specializing them to production SHA-256.

## Core Principle

Keep these roles separate:

- `Shape`: static relation metadata and layout.
- `Witness`: the prover's full pre-projection assignment.
- `Instance`: verifier-visible public data plus commitments.
- `FoldedWitness`: prover-private folded state after folding.
- `FoldedInstance`: verifier-visible folded accumulator.

Do not put projected/evaluated data in the initial instance object. Projection
and evaluation happen later after transcript challenges are known.

## Existing e2e.rs Lifecycle

The generic path in `protocol/benches/e2e.rs` works like this:

```rust
let trace = U::generate_random_trace(num_vars, &mut rng);
let sig = U::signature();
let public_trace = trace.public(&sig);
```

The prover receives the full `UairTrace`, splits it with `UairSignature`, commits
only the witness columns, and absorbs the unprojected public columns into the
transcript:

```rust
let public_trace = trace.public(&uair_signature);
let witness_trace = trace.witness(&uair_signature);

commit(witness_trace);
absorb(public_trace);
```

The verifier receives the proof and the unprojected `public_trace`. It absorbs
that same public trace before later projection/evaluation steps.

So the instance's public data is an unprojected public UAIR trace, not a
`ProjectedPublicTrace`.

## UairSignature

`UairSignature` is the layout contract for a UAIR. It defines:

- total column counts for `binary_poly`, `arbitrary_poly`, and `int`
- public column prefix counts
- witness column suffix counts
- shifted columns
- virtual columns
- lookup and booleanity metadata

`UairTrace::public(sig)` and `UairTrace::witness(sig)` use this signature to
split a full trace into public and witness subtraces.

## Generic Objects

`UairShape` is useful as a value-level handle for the static UAIR relation plus
the trace length.

```rust
pub struct UairShape<U: Uair> {
    pub num_vars: usize,
    pub signature: UairSignature,
    _marker: PhantomData<U>,
}
```

`num_vars` should stay on the shape or protocol input. It is the log trace
length, so the row domain has size `1 << num_vars`. The prover and verifier use
it for MLE sizes, sumcheck rounds, public-structure checks, and PCS parameters.

Do not add `shape_digest` in the first pass. A shape digest may be useful later
for serialization or cached transcript binding, but it should be derived from
the shape rather than treated as fundamental state.

The witness is the full pre-projection prover assignment. It includes public
columns and private columns because `UairTrace` itself stores both; public
columns are the prefix determined by `UairSignature`.

```rust
pub struct UairWitness<'a, PolyCoeff: Clone, Int: Clone, const D: usize> {
    pub trace: UairTrace<'a, PolyCoeff, Int, D>,
}
```

Document this clearly: `UairWitness` means full prover assignment, not
private-only data.

The fresh verifier-visible instance contains the unprojected public trace and
commitments to the witness columns.

```rust
pub struct UairInstance<'a, PolyCoeff: Clone, Int: Clone, Commitments, const D: usize> {
    pub public_trace: UairTrace<'a, PolyCoeff, Int, D>,
    pub commitments: Commitments,
}
```

The folded objects should mirror the fresh split:

```rust
pub struct FoldedUairWitness<F> {
    // prover-private folded evaluations, residuals, and randomness
}

pub struct FoldedUairInstance<F, Commitments> {
    pub commitments: Commitments,
    pub public_evals: Vec<F>,
    pub u: F,
}
```

The exact fields of the folded objects should be chosen by the folding protocol,
but the boundary remains the same: witness is private, instance is
verifier-visible.

## LinearIdealFold Proof Objects

`LinearIdealFold` is the generic folding layer for UAIRs whose projected
residue constraints live in linear ideals. The proof object should contain only
verifier messages and claimed evaluations. It should not contain prover-side
caches such as PCS prover data, and it should not contain NeutronNova-specific
objects such as `comm_E` for a committed power-vector witness.

The family association for ideal polynomials is part of the proof contract. A
bare `Vec<DynamicPolynomialF<F>>` is not enough unless `UairShape` defines the
exact canonical order. Prefer an explicit family-tagged form:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IdealFamilyId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IdealPolySlot(pub u16);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdealFamilyPolys<F: PrimeField> {
    pub family: IdealFamilyId,
    pub polys: Vec<IdealFamilyPoly<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdealFamilyPoly<F: PrimeField> {
    pub slot: IdealPolySlot,
    pub poly: DynamicPolynomialF<F>,
}
```

The verifier must check that these families and slots are in the canonical
shape-defined order, with no missing or duplicate entries. For SHA-256, this
corresponds to the current `production_sha_nonzero_families()` order, but the
generic API should make that association shape-level data.

The folded verifier-visible instance is the result of folding fresh instances.
It contains the folded target claim, folded commitments, and any folded public
values needed by the later checks:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedLinearIdealInstance<F: PrimeField, Commitments, Public> {
    pub target: F,
    pub commitments: Commitments,
    pub public: Public,
}
```

The folded witness is prover-private. Its concrete representation can be an
owned folded trace, folded source MLEs, or another protocol-specific witness
bundle:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedLinearIdealWitness<Witness> {
    pub witness: Witness,
}
```

The proof object is:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearIdealFoldProof<F: PrimeField> {
    pub ideal_family_polys: Vec<IdealFamilyPolys<F>>,
    pub nifs: MultiDegreeSumcheckProof<F>,
    pub folded_claim_sumcheck: MultiDegreeSumcheckProof<F>,
    pub terminal_evals: TerminalEvals<F>,
    pub multipoint: MultipointEvalProof<F>,
    pub opening_evals: OpeningEvals<F>,
    pub pcs_opening_bytes: Vec<u8>,
}
```

`ideal_family_polys` contains the nonzero ideal-witness polynomials, grouped by
ideal family. These prove ideal membership and define the scalar targets used by
the instance-axis folding claim.

For fresh instances \(b \in \{0,\dots,B-1\}\), ideal families
\(f \in \mathcal F\), and family slots \(k\):

$$
E_{b,f,k}(X) \in I_f
$$

The beta-aggregated family polynomial is:

$$
\bar E^{\beta}_{f,k}(X)
=
\sum_{b=0}^{B-1} \operatorname{eq}(\beta,b)\,E_{b,f,k}(X)
$$

Because the ideals are linear:

$$
E_{b,f,k}(X) \in I_f
\quad\Longrightarrow\quad
\bar E^{\beta}_{f,k}(X) \in I_f
$$

The fresh scalar target for instance \(b\) is:

$$
T_b
=
\sum_{f \in \mathcal F}
\lambda_f
\sum_k E_{b,f,k}(a)
$$

The initial SumFold claim is:

$$
C_0
=
\sum_{b=0}^{B-1}
\operatorname{eq}(\beta,b)\,T_b
$$

`nifs` is the instance-axis `MultiDegreeSumcheckProof`. It proves the SumFold
transition from \(C_0\) to the folded target. The verifier derives \(r_b\),
folding weights \(\theta_b\), and \(T'\):

$$
\theta_b = \operatorname{eq}(r_b,b)
$$

$$
T'
=
\frac{c_{\mathrm{SF}}}{\operatorname{eq}(\beta,r_b)}
$$

`folded_claim_sumcheck` proves that the folded target is the row-domain sum of
the folded residue expression:

$$
T'
=
\sum_{x \in \{0,1\}^{d}}
\operatorname{eq}(r_{\mathrm{ic}},x)
\cdot
\Phi_{\mathrm{folded}}(x)
$$

This sumcheck reduces the folded claim to a terminal point \(r_\star\).

`terminal_evals` contains the claimed evaluations needed to reconstruct
\(\Phi_{\mathrm{folded}}(r_\star)\). It should include source identifiers,
shift identifiers, scalarized values, and any coefficient-level values needed by
the shape-specific expression. For SHA-256 this corresponds to the current
endpoint evaluations of folded word and integer sources.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShiftId(pub u16);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalEvals<F: PrimeField> {
    pub polynomial_sources: Vec<TerminalPolynomialEval<F>>,
    pub scalar_sources: Vec<TerminalScalarEval<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalPolynomialEval<F: PrimeField> {
    pub source: SourceId,
    pub shift: ShiftId,
    pub scalarized: F,
    pub coeffs: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalScalarEval<F: PrimeField> {
    pub source: SourceId,
    pub shift: ShiftId,
    pub value: F,
}
```

The verifier uses `terminal_evals` to check:

$$
\mathrm{terminal}
=
\operatorname{eq}(r_{\mathrm{ic}},r_\star)
\cdot
\Phi_{\mathrm{folded}}(r_\star)
$$

`multipoint` reduces all terminal evaluation claims at \(r_\star\) and shifted
points into one batched opening claim at a verifier-derived point \(r_0\):

$$
\{p_i(s_i(r_\star)) = v_i\}_i
\quad\Longrightarrow\quad
P(r_0)=v_0
$$

`opening_evals` are the claimed folded committed-source evaluations at \(r_0\).
`pcs_opening_bytes` is the serialized PCS proof that those evaluations match
the folded commitments.

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningEvals<F: PrimeField> {
    pub polynomial_sources: Vec<DynamicPolynomialF<F>>,
    pub scalar_sources: Vec<F>,
}
```

The proof chain is:

```text
ideal_family_polys
    -> derive C0 and prove ideal membership
nifs
    -> fold C0 into T'
folded_claim_sumcheck
    -> reduce T' to terminal point r_star
terminal_evals
    -> reconstruct the terminal folded expression
multipoint
    -> reduce many endpoint claims to one opening point r_0
opening_evals + pcs_opening_bytes
    -> prove consistency with folded commitments
```

## SHA-256 Domain Objects

For one SHA-256 compression, the semantic relation is:

```rust
H_{i+1} = compress(H_i, M_i)
```

where:

```rust
H_i:     [u32; 8]
M_i:     [u32; 16]
H_{i+1}: [u32; 8]
```

For a chain of `N` compressions:

```rust
pub struct Sha256ChainPublicInput<const N: usize> {
    pub initial_state: [u32; 8],
    pub message_blocks: [[u32; 16]; N],
    pub final_state: [u32; 8],
}
```

For standard SHA-256 hashing from the fixed IV, use a wrapper:

```rust
pub struct Sha256HashPublicInput<const N: usize> {
    pub message_blocks: [[u32; 16]; N],
    pub digest: [u32; 8],
}
```

This wrapper expands into `Sha256ChainPublicInput` by setting
`initial_state = SHA256_IV`.

The current e2e SHA UAIR packs multiple compressions into a single trace. It
does not pass intermediate states as separate public inputs. Witness generation
computes:

```rust
H_1 = compress(H_0, M_0)
H_2 = compress(H_1, M_1)
...
H_N = compress(H_{N-1}, M_{N-1})
```

and writes the relevant public values into public UAIR columns.

## SHA-256 to UAIR Mapping

The SHA domain input is not the same thing as the UAIR public trace.

The SHA public input:

```rust
Sha256ChainPublicInput {
    initial_state,
    message_blocks,
    final_state,
}
```

is used to build the UAIR public trace. For the current SHA UAIR this includes
columns such as:

- `PA_M`: message block words
- `PA_A` / `PA_E`: chaining states and final output prefix
- `PA_K`: SHA-256 round constants
- selector columns
- implementation-specific public helper columns

The prover also builds the full `UairWitness` trace containing the public
columns plus private/witness columns such as:

- message schedule `W`
- round state columns
- sigma/Sigma columns
- Ch/Maj auxiliary columns
- carry columns
- compensator columns

The verifier should build or receive only the public trace, then verify the
proof against commitments to the private columns.

## Recommended Production Flow

Witness generation is outside the prover. It consumes semantic public input and
produces a full pre-projection UAIR witness:

```rust
pub fn build_uair_witness<U, Input, PolyCoeff, Int, const D: usize>(
    shape: &UairShape<U>,
    public: &Input,
) -> Result<UairWitness<'static, PolyCoeff, Int, D>, UairWitnessError>
where
    U: Uair,
    PolyCoeff: Clone,
    Int: Clone;
```

The SHA-256 instantiation can be written more concretely as:

```rust
pub fn build_sha256_witness<Zt, const N: usize, const D: usize>(
    shape: &UairShape<ProductionShaUair<Zt::Int, N>>,
    public: &Sha256ChainPublicInput<N>,
) -> Result<UairWitness<'static, Zt::Int, Zt::Int, D>, ShaWitnessError>
where
    Zt: ZincTypes<D>;
```

The prover receives witnesses and commits to the witness columns internally. It
returns fresh verifier-visible instances, the folded accumulator pair, and the
proof:

```rust
pub struct LinearIdealFoldProveOutput<Instance, FoldedInstance, FoldedWitness, Proof> {
    pub fresh_instances: Vec<Instance>,
    pub folded_instance: FoldedInstance,
    pub folded_witness: FoldedWitness,
    pub proof: Proof,
}

pub fn prove_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    pp: &LinearIdealFoldProverParams<P, U, Zt, F, D>,
    shape: &UairShape<U>,
    witnesses: &[UairWitness<'_, Zt::Int, Zt::Int, D>],
    transcript: &mut impl Transcript,
) -> Result<
    LinearIdealFoldProveOutput<
        UairInstance<'static, Zt::Int, Zt::Int, PCSCommitments<P, Zt, F, D>, D>,
        FoldedLinearIdealInstance<F, PCSCommitments<P, Zt, F, D>, FoldedPublicEvals<F>>,
        FoldedLinearIdealWitness<FoldedUairTrace<F>>,
        LinearIdealFoldProof<F>,
    >,
    LinearIdealFoldError<F>,
>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField;
```

The verifier receives the fresh instances and proof. It derives the same folded
instance, but it never receives the folded witness:

```rust
pub fn verify_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    vp: &LinearIdealFoldVerifierParams<P, U, Zt, F, D>,
    shape: &UairShape<U>,
    instances: &[UairInstance<'_, Zt::Int, Zt::Int, PCSCommitments<P, Zt, F, D>, D>],
    proof: &LinearIdealFoldProof<F>,
    transcript: &mut impl Transcript,
) -> Result<
    FoldedLinearIdealInstance<F, PCSCommitments<P, Zt, F, D>, FoldedPublicEvals<F>>,
    LinearIdealFoldError<F>,
>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField;
```

```rust
Sha256ChainPublicInput
    -> build public UairTrace

Sha256ChainPublicInput + witness generation
    -> full UairWitness

UairWitness + UairShape
    -> commit witness columns
    -> UairInstance { public_trace, commitments }
    -> prove

UairInstance + proof
    -> verify
```

Projection to the proof field and evaluation at verifier challenges are internal
protocol phases. They should not be part of the initial public instance type.
