//! Phase C — the Binius64 **oblong AND zerocheck** wired into the F_2 Hadamard
//! discharge for a single AND relation `U ⊙ V = W`, with the **`ψ_z`
//! recombination tie** that binds the zerocheck's operand evals to the
//! committed columns (port plan `documentation/f2-hadamard-oblong-port-plan.md`
//! §4).
//!
//! ## The seam, in one paragraph
//!
//! The oblong zerocheck ([`zinc_poly::univariate::oblong_and`]) outputs
//! `a_eval = ψ_z(U)(γ)`, `b_eval = ψ_z(V)(γ)`, `c_eval = ψ_z(W)(γ)` at the
//! oblong point `(z, γ)` — where `ψ_z(W) = Σ_i W_i·L_i(z)` collapses a word's
//! `D` bits into one `GF(2^128)` scalar using the additive-NTT Lagrange basis
//! `L_i` evaluated at `z`. This is the exact analogue of today's
//! `ψ_α(W) = Σ_b W_b·α^b` (monomial basis), so the tie **reuses the existing
//! `ψ_α` machinery** ([`pair_alpha_evals`] + [`derive_operand_parents`]) fed the
//! Lagrange weights `base_lagrange_at(z)` in place of the α-powers and the
//! Phase-2 sumcheck point `γ` in place of `r*`. `ψ_z` is `F_2`-linear, so it
//! commutes with the XOR/shift/complement operand structure exactly as `ψ_α`
//! does; the derived operand evals must equal the zerocheck's `a/b/c_eval`.
//! Soundness is the same `(D−1)/|F|` Schwartz–Zippel bound as `ψ_α` (plan §4).
//!
//! ## Status
//!
//! One AND relation, **explicit challenges** (`r`, `z`, `γ`) — the Fiat–Shamir
//! transcript wiring and the multi-relation batching are the follow-ups (plan
//! §5, Phase D). The tie does *not* yet bind the operand evals to the
//! commitment; it checks the derived evals against the in-memory columns, which
//! validates the recombination algebra (the architectural risk).
//!
//! **The binding is a row-space opening at `γ`, not a joint `(z, γ)` open.**
//! `z` is the *bit-recombination*: it selects the weights `L_i(z)` that collapse
//! the `D` bit-slices into one `GF(2^128)` scalar — exactly as `α` (via `α^b`)
//! does in today's discharge — so the opened object is the `ψ_z`-projected
//! column, a row-multilinear opened at the row point `γ`. The remaining wiring
//! is therefore to **fold the `ψ_z(col↓Δ)(γ)` pair-evals into the main
//! multipoint-eval** (Δ=0 as a point claim, Δ≠0 via the shift predicate, bound
//! by the single PCS open) — the same "Approach B" the `ψ_α` pair-evals already
//! use (`f2_prove.rs`), with `L_i(z)`/`γ` for `α^b`/`r*`.

use crypto_primitives::Field;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;
use zinc_poly::univariate::binary_subspace::BinarySubspace;
use zinc_poly::univariate::oblong_and::{
    AdditiveNtt, AndCheckOutput, MonomialScheme, OblongAndProof, OblongChannel, OblongError,
    OblongScheme, SKIPPED_VARS, WORD_BITS, base_lagrange_at, eq_indicator, prove_oblong_and_channel,
    verify_oblong_and_channel,
};
use zinc_poly::univariate::oblong_and_gf8::Gf8Scheme;
use zinc_transcript::traits::{ConstTranscribable, Transcript};

use crate::f2_hadamard::{
    F2AdderSpec, F2HadamardSpec, F2Operand, adder_operand_alpha_evals, build_adder_operand_columns,
    build_operand_column, cell_mask, derive_operand_parents, distinct_pairs, pair_alpha_evals,
};

type Gf = BinaryFieldGF128;

/// [`OblongChannel`] adapter over a Fiat–Shamir [`Transcript`]: absorbs prover
/// messages and samples `GF(2^128)` challenges. The transcript carries the
/// challenges, so prover and verifier (mirroring the same absorb/sample order)
/// agree without shipping `z`/`γ` in the proof.
struct TranscriptChannel<'a, T: Transcript> {
    transcript: &'a mut T,
    buf: Vec<u8>,
}

impl<'a, T: Transcript> TranscriptChannel<'a, T> {
    fn new(transcript: &'a mut T) -> Self {
        let n_bytes = <<Gf as Field>::Inner as ConstTranscribable>::NUM_BYTES;
        Self {
            transcript,
            buf: vec![0u8; n_bytes],
        }
    }
}

impl<'a, T: Transcript> OblongChannel for TranscriptChannel<'a, T> {
    fn absorb(&mut self, scalars: &[Gf]) {
        self.transcript.absorb_random_field_slice(scalars, &mut self.buf);
    }
    fn sample(&mut self) -> Gf {
        self.transcript.get_field_challenge(&())
    }
}

/// SHA-256 word width — the oblong "Z" (bit-index) dimension. The oblong port
/// is specialised to `D = 32` (`WORD_BITS`).
const D: usize = WORD_BITS;

/// Errors from [`verify_oblong_and_relation`].
#[derive(Debug, thiserror::Error)]
pub enum OblongVerifyError {
    #[error("oblong zerocheck rejected: {0}")]
    Oblong(OblongError),
    #[error("ψ_z tie failed: derived operand eval(s) ≠ zerocheck eval(s)")]
    TieMismatch,
}

/// Pack a built operand column into `u32` words (one per row), the oblong
/// prover's input. Reuses [`build_operand_column`] so the operand semantics
/// (XOR of `↓Δ`-shifted columns, optional complement) match the `ψ_z` tie.
fn operand_words(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    operand: &F2Operand,
    num_vars: usize,
) -> Vec<u32> {
    let col = build_operand_column::<D>(columns, operand, num_vars);
    col.evaluations
        .iter()
        .map(|c| cell_mask::<D>(c) as u32)
        .collect()
}

/// Prover: run the oblong AND zerocheck for one relation `U ⊙ V = W` over the
/// committed columns, with Fiat–Shamir challenges drawn from `transcript`.
pub fn prove_oblong_and_relation<T: Transcript>(
    transcript: &mut T,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    spec: &F2HadamardSpec,
    num_vars: usize,
) -> OblongAndProof {
    let u = operand_words(columns, &spec.u, num_vars);
    let v = operand_words(columns, &spec.v, num_vars);
    let w = operand_words(columns, &spec.w, num_vars);
    let ntt = AdditiveNtt::new();
    let scheme = MonomialScheme::new(&ntt);
    let mut ch = TranscriptChannel::new(transcript);
    prove_oblong_and_channel(&mut ch, &u, &v, &w, &scheme).0
}

/// Verifier: re-derive the challenges from `transcript`, check the oblong
/// zerocheck, then tie its operand evals to the committed columns via the
/// `ψ_z` recombination (the existing `ψ_α` machinery with `base_lagrange_at(z)`
/// weights and the row-point `γ`, from the verified `eval_point = [z, γ…]`).
pub fn verify_oblong_and_relation<T: Transcript>(
    transcript: &mut T,
    proof: &OblongAndProof,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    spec: &F2HadamardSpec,
    num_vars: usize,
) -> Result<(), OblongVerifyError> {
    let full = BinarySubspace::with_dim(SKIPPED_VARS + 1);
    let out: AndCheckOutput = {
        let mut ch = TranscriptChannel::new(transcript);
        verify_oblong_and_channel(&mut ch, proof, num_vars, &full, &[])
            .map_err(OblongVerifyError::Oblong)?
    };

    // ψ_z recombination. eval_point = [z, γ_0, …, γ_{n-1}]; the tie evaluates at
    // the row-point γ. Feed the Lagrange weights L_i(z) where ψ_α uses α^b.
    let z = out.eval_point[0];
    let gamma = &out.eval_point[1..];
    let lagrange_z = base_lagrange_at(z).to_vec();
    let specs = std::slice::from_ref(spec);
    let pairs = distinct_pairs(specs);
    let pair_evals = pair_alpha_evals::<D>(columns, &pairs, &lagrange_z, gamma);
    let parents = derive_operand_parents(specs, &pairs, &pair_evals, &lagrange_z);

    // parents = [ψ_z(U)(γ), ψ_z(V)(γ), ψ_z(W)(γ)] must equal the zerocheck evals.
    if parents.len() != 3
        || parents[0] != out.a_eval
        || parents[1] != out.b_eval
        || parents[2] != out.c_eval
    {
        return Err(OblongVerifyError::TieMismatch);
    }
    Ok(())
}

/// Prover for a **batch** of `K` AND relations: stack their operand columns into
/// one oblong zerocheck. The relation index forms the high row variables, the
/// word index the low ones, padded to a power-of-two relation count with
/// zero-operand relations (`0 ⊙ 0 = 0`, vacuously satisfied). One univariate-skip
/// round + one Phase-2 sumcheck cover all relations (Binius's stacking — vs the
/// current discharge's `γ`-batch).
/// Convert a built `BinaryPoly<D>` operand column to `u32` words.
fn words_of(col: &DenseMultilinearExtension<BinaryPoly<D>>) -> Vec<u32> {
    col.evaluations.iter().map(|c| cell_mask::<D>(c) as u32).collect()
}

/// Build the stacked `(a, b, c)` operand-word columns for all relations (ANDs
/// first, then adders in Binius carry-AND form), padded to a power-of-two
/// relation count with zero-operand rows. The stack order is the relation index;
/// the ψ_z tie derives evals in the same order.
#[allow(clippy::arithmetic_side_effects)]
fn build_stacked_operands(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    and_specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = 1usize << num_vars;
    let k = and_specs.len() + adder_specs.len();
    let k_pad = k.next_power_of_two().max(1);
    let mut a = Vec::with_capacity(k_pad * n);
    let mut b = Vec::with_capacity(k_pad * n);
    let mut c = Vec::with_capacity(k_pad * n);
    for spec in and_specs {
        a.extend(operand_words(columns, &spec.u, num_vars));
        b.extend(operand_words(columns, &spec.v, num_vars));
        c.extend(operand_words(columns, &spec.w, num_vars));
    }
    for adder in adder_specs {
        let uvw = build_adder_operand_columns::<D>(columns, adder, num_vars);
        a.extend(words_of(&uvw[0]));
        b.extend(words_of(&uvw[1]));
        c.extend(words_of(&uvw[2]));
    }
    a.resize(k_pad * n, 0);
    b.resize(k_pad * n, 0);
    c.resize(k_pad * n, 0);
    (a, b, c)
}

/// Stacked row-variable count for `k` relations over `num_vars`-row columns.
#[allow(clippy::arithmetic_side_effects)]
fn stacked_nvars(k: usize, num_vars: usize) -> usize {
    num_vars + k.next_power_of_two().max(1).trailing_zeros() as usize
}

/// The batched ψ_z tie: `a/b/c_eval = Σ_rel eq(rel; γ_rel)·ψ_z(operand_rel)(γ_word)`,
/// `γ = [γ_word (low num_vars), γ_rel (high log k_pad)]`. `lagrange_z` are the
/// base-domain Lagrange weights `L_b(z)` for the chosen scheme (monomial or
/// `embed(H₈)`), matching how the prover folded the operands. ANDs derive from
/// the committed columns via the pair evals; adders recompute the carry operands.
#[allow(clippy::arithmetic_side_effects)]
fn batched_tie_check(
    out: &AndCheckOutput,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    and_specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
    lagrange_z: &[Gf],
) -> Result<(), OblongVerifyError> {
    let k = and_specs.len() + adder_specs.len();
    let gammas = &out.eval_point[1..];
    let (gamma_word, gamma_rel) = gammas.split_at(num_vars);

    let pairs = distinct_pairs(and_specs);
    let pair_evals = pair_alpha_evals::<D>(columns, &pairs, lagrange_z, gamma_word);
    let mut parents = derive_operand_parents(and_specs, &pairs, &pair_evals, lagrange_z);
    for adder in adder_specs {
        let uvw = build_adder_operand_columns::<D>(columns, adder, num_vars);
        parents.extend(adder_operand_alpha_evals::<D>(&uvw, lagrange_z, gamma_word));
    }

    let eq_rel = eq_indicator(gamma_rel);
    let mut exp = [Gf::zero(); 3];
    for (rel, &w) in eq_rel.iter().take(k).enumerate() {
        for j in 0..3 {
            exp[j] = exp[j] + w * parents[3 * rel + j];
        }
    }
    if exp[0] != out.a_eval || exp[1] != out.b_eval || exp[2] != out.c_eval {
        return Err(OblongVerifyError::TieMismatch);
    }
    Ok(())
}

/// Batched oblong discharge prover, **naive `GF(2^128)`** (monomial scheme).
pub fn prove_oblong_and_batch<T: Transcript>(
    transcript: &mut T,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    and_specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
) -> OblongAndProof {
    let (a, b, c) = build_stacked_operands(columns, and_specs, adder_specs, num_vars);
    let ntt = AdditiveNtt::new();
    let scheme = MonomialScheme::new(&ntt);
    let mut ch = TranscriptChannel::new(transcript);
    prove_oblong_and_channel(&mut ch, &a, &b, &c, &scheme).0
}

/// **`GF(2^8)`-accelerated** batched oblong discharge prover: same stacked
/// zerocheck, but the additive-NTT + per-extension-point products run in
/// `GF(2^8)` over `embed(H₈)` (the byte-lookup scheme).
///
/// Returns the proof **and** the eval-point `[z, γ…]` (`γ` splits into `γ_word`
/// over the low `num_vars` row variables + `γ_rel` over the high `log2 k_pad`
/// relation variables). The sound binding folds the `ψ_z(col↓Δ)(γ_word)`
/// pair-evals into the main multipoint-eval, so it needs `z`/`γ_word`; the
/// standalone discharge bench drops the point.
pub fn prove_oblong_and_batch_gf8<T: Transcript>(
    transcript: &mut T,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    and_specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
) -> (OblongAndProof, Vec<Gf>) {
    let (a, b, c) = build_stacked_operands(columns, and_specs, adder_specs, num_vars);
    let scheme = Gf8Scheme::new();
    let mut ch = TranscriptChannel::new(transcript);
    prove_oblong_and_channel(&mut ch, &a, &b, &c, &scheme)
}

/// Verify the batched oblong zerocheck and tie **every** relation's operand
/// evals to the committed columns. The eval point is `[z, γ]` with `γ` splitting
/// into `γ_word` (low `num_vars`) + `γ_rel` (high `log2 k_pad`); the batched tie
/// checks `a_eval = Σ_rel eq(rel; γ_rel)·ψ_z(U_rel)(γ_word)` and likewise b/c.
/// Padding relations have zero operands, so they contribute zero and are skipped.
/// Verify the batched oblong discharge, **naive `GF(2^128)`** (monomial scheme):
/// `R₀` reconstructed over the monomial dim-`(SKIPPED_VARS+1)` subspace, tie
/// weights `L_b(z)` over the monomial base subspace.
pub fn verify_oblong_and_batch<T: Transcript>(
    transcript: &mut T,
    proof: &OblongAndProof,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    and_specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
) -> Result<(), OblongVerifyError> {
    let k = and_specs.len() + adder_specs.len();
    let full = BinarySubspace::with_dim(SKIPPED_VARS + 1);
    let out = {
        let mut ch = TranscriptChannel::new(transcript);
        verify_oblong_and_channel(&mut ch, proof, stacked_nvars(k, num_vars), &full, &[])
            .map_err(OblongVerifyError::Oblong)?
    };
    let lagrange_z = base_lagrange_at(out.eval_point[0]).to_vec();
    batched_tie_check(&out, columns, and_specs, adder_specs, num_vars, &lagrange_z)
}

/// Verify the **`GF(2^8)`-accelerated** batched oblong discharge: `R₀`
/// reconstructed over `embed(H₈)`, tie weights `L_b(z)` over the `embed(H₈)`
/// base subspace (matching [`prove_oblong_and_batch_gf8`]).
pub fn verify_oblong_and_batch_gf8<T: Transcript>(
    transcript: &mut T,
    proof: &OblongAndProof,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    and_specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
) -> Result<(), OblongVerifyError> {
    let k = and_specs.len() + adder_specs.len();
    let scheme = Gf8Scheme::new();
    let out = {
        let mut ch = TranscriptChannel::new(transcript);
        verify_oblong_and_channel(
            &mut ch,
            proof,
            stacked_nvars(k, num_vars),
            scheme.full_subspace(),
            scheme.small_challenges(),
        )
        .map_err(OblongVerifyError::Oblong)?
    };
    let lagrange_z = scheme.base_lagrange(out.eval_point[0]).to_vec();
    batched_tie_check(&out, columns, and_specs, adder_specs, num_vars, &lagrange_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::boolean::Boolean;
    use zinc_transcript::Blake3Transcript;

    fn col_from_u32s(patterns: &[u32]) -> DenseMultilinearExtension<BinaryPoly<D>> {
        use std::array;
        let evaluations: Vec<BinaryPoly<D>> = patterns
            .iter()
            .map(|&p| {
                let coeffs: [Boolean; D] = array::from_fn(|i| Boolean::new((p >> i) & 1 != 0));
                BinaryPoly::<D>::new(coeffs)
            })
            .collect();
        let num_vars = patterns.len().next_power_of_two().trailing_zeros() as usize;
        DenseMultilinearExtension {
            num_vars,
            evaluations,
        }
    }

    /// Honest oblong prove → verify (zerocheck + ψ_z tie) for the given spec,
    /// with real Fiat–Shamir transcripts (fresh, same-seed) on each side.
    fn round_trip(columns: &[DenseMultilinearExtension<BinaryPoly<D>>], spec: &F2HadamardSpec, num_vars: usize) {
        let mut tp = Blake3Transcript::new();
        let proof = prove_oblong_and_relation(&mut tp, columns, spec, num_vars);
        let mut tv = Blake3Transcript::new();
        verify_oblong_and_relation(&mut tv, &proof, columns, spec, num_vars)
            .expect("honest oblong round-trip + ψ_z tie");
    }

    // 8 rows of 32-bit words (num_vars = 3).
    const U: [u32; 8] = [
        0xDEAD_BEEF, 0x0BAD_F00D, 0x1234_5678, 0x9ABC_DEF0, 0xFFFF_0000, 0x00FF_00FF, 0xA5A5_A5A5, 0x5A5A_5A5A,
    ];
    const V: [u32; 8] = [
        0xCAFE_BABE, 0xFEED_FACE, 0x8765_4321, 0x0FED_CBA9, 0x0F0F_F0F0, 0xFF00_FF00, 0x3333_CCCC, 0xCCCC_3333,
    ];

    #[test]
    fn plain_and_round_trips() {
        let w: Vec<u32> = U.iter().zip(&V).map(|(a, b)| a & b).collect();
        let columns = [col_from_u32s(&U), col_from_u32s(&V), col_from_u32s(&w)];
        round_trip(&columns, &F2HadamardSpec::plain(0, 1, 2), 3);
    }

    #[test]
    fn row_shift_and_round_trips() {
        // W = A ⊙ A^↓1 (row t: A[t] & A[t+1], A[8]=0).
        let n = 8usize;
        let w: Vec<u32> = (0..n).map(|t| if t + 1 < n { U[t] & U[t + 1] } else { 0 }).collect();
        let columns = [col_from_u32s(&U), col_from_u32s(&w)];
        let spec = F2HadamardSpec {
            u: F2Operand::col(0),
            v: F2Operand::shifted(0, 1),
            w: F2Operand::col(1),
        };
        round_trip(&columns, &spec, 3);
    }

    #[test]
    fn complement_and_round_trips() {
        // W = (1 − A) ⊙ B.
        let w: Vec<u32> = U.iter().zip(&V).map(|(a, b)| (!a) & b).collect();
        let columns = [col_from_u32s(&U), col_from_u32s(&V), col_from_u32s(&w)];
        let spec = F2HadamardSpec {
            u: F2Operand::col(0).complemented(),
            v: F2Operand::col(1),
            w: F2Operand::col(2),
        };
        round_trip(&columns, &spec, 3);
    }

    #[test]
    fn maj_combo_round_trips() {
        // C14 shape: (A⊕A^↓2) ⊙ (A^↓1⊕A^↓2) = (W_col⊕A^↓2), W_col = (U⊙V)⊕A^↓2.
        let n = 8usize;
        let sh = |arr: &[u32], d: usize, t: usize| if t + d < n { arr[t + d] } else { 0 };
        let w_col: Vec<u32> = (0..n)
            .map(|t| {
                let u = U[t] ^ sh(&U, 2, t);
                let v = sh(&U, 1, t) ^ sh(&U, 2, t);
                (u & v) ^ sh(&U, 2, t)
            })
            .collect();
        let columns = [col_from_u32s(&U), col_from_u32s(&w_col)];
        let spec = F2HadamardSpec {
            u: F2Operand::xor(vec![
                crate::f2_hadamard::F2OperandTerm { col: 0, row_shift: 0 },
                crate::f2_hadamard::F2OperandTerm { col: 0, row_shift: 2 },
            ]),
            v: F2Operand::xor(vec![
                crate::f2_hadamard::F2OperandTerm { col: 0, row_shift: 1 },
                crate::f2_hadamard::F2OperandTerm { col: 0, row_shift: 2 },
            ]),
            w: F2Operand::xor(vec![
                crate::f2_hadamard::F2OperandTerm { col: 1, row_shift: 0 },
                crate::f2_hadamard::F2OperandTerm { col: 0, row_shift: 2 },
            ]),
        };
        round_trip(&columns, &spec, 3);
    }

    #[test]
    fn corrupt_w_is_rejected() {
        // Corrupt one W word ⇒ the AND fails ⇒ R₀ no longer vanishes on the base
        // domain ⇒ the reconstructed claim is wrong. The eq-factored MLE-check
        // threads that bad claim to the closing check, which rejects.
        let mut w: Vec<u32> = U.iter().zip(&V).map(|(a, b)| a & b).collect();
        w[2] ^= 1 << 7;
        let columns = [col_from_u32s(&U), col_from_u32s(&V), col_from_u32s(&w)];
        let spec = F2HadamardSpec::plain(0, 1, 2);
        let num_vars = 3;
        let mut tp = Blake3Transcript::new();
        let proof = prove_oblong_and_relation(&mut tp, &columns, &spec, num_vars);
        let mut tv = Blake3Transcript::new();
        let res = verify_oblong_and_relation(&mut tv, &proof, &columns, &spec, num_vars);
        assert!(
            matches!(res, Err(OblongVerifyError::Oblong(OblongError::FinalCheck))),
            "corrupt W must be rejected by the zerocheck, got {res:?}"
        );
    }

    #[test]
    fn tie_catches_wrong_operand_wiring() {
        // The zerocheck passes for an honest W=U⊙V, but if the verifier ties to
        // the WRONG columns (swap U/V source vs the proven operands), the ψ_z
        // tie must reject. Here we prove U⊙V=W honestly, then verify against a
        // spec whose W operand points at the wrong column.
        let w: Vec<u32> = U.iter().zip(&V).map(|(a, b)| a & b).collect();
        let columns = [col_from_u32s(&U), col_from_u32s(&V), col_from_u32s(&w)];
        let num_vars = 3;
        let mut tp = Blake3Transcript::new();
        let proof = prove_oblong_and_relation(&mut tp, &columns, &F2HadamardSpec::plain(0, 1, 2), num_vars);
        // Verify with W tied to column 0 (= U) instead of column 2 (= W). The
        // zerocheck still passes (same proof + transcript); the ψ_z tie rejects.
        let bad_spec = F2HadamardSpec::plain(0, 1, 0);
        let mut tv = Blake3Transcript::new();
        let res = verify_oblong_and_relation(&mut tv, &proof, &columns, &bad_spec, num_vars);
        assert!(matches!(res, Err(OblongVerifyError::TieMismatch)), "got {res:?}");
    }

    fn round_trip_batch(
        columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
        and_specs: &[F2HadamardSpec],
        adder_specs: &[F2AdderSpec],
        num_vars: usize,
    ) {
        let mut tp = Blake3Transcript::new();
        let proof = prove_oblong_and_batch(&mut tp, columns, and_specs, adder_specs, num_vars);
        let mut tv = Blake3Transcript::new();
        verify_oblong_and_batch(&mut tv, &proof, columns, and_specs, adder_specs, num_vars)
            .expect("batched oblong round-trip + ψ_z tie");
    }

    #[test]
    fn batch_three_and_relations_round_trips() {
        // rel0: U⊙V=W0 (cols 0,1,2); rel1: V⊙P=W1 (cols 3,4,5);
        // rel2: U⊙U↓1=W2 (cols 6,7). K=3 ⇒ k_pad=4 (one zero-operand padding relation).
        let n = 8usize;
        let p: Vec<u32> = U.iter().map(|x| x.rotate_left(5) ^ 0x55).collect();
        let w0: Vec<u32> = U.iter().zip(&V).map(|(x, y)| x & y).collect();
        let w1: Vec<u32> = V.iter().zip(&p).map(|(x, y)| x & y).collect();
        let w2: Vec<u32> = (0..n).map(|t| if t + 1 < n { U[t] & U[t + 1] } else { 0 }).collect();
        let columns = [
            col_from_u32s(&U),
            col_from_u32s(&V),
            col_from_u32s(&w0),
            col_from_u32s(&V),
            col_from_u32s(&p),
            col_from_u32s(&w1),
            col_from_u32s(&U),
            col_from_u32s(&w2),
        ];
        let specs = [
            F2HadamardSpec::plain(0, 1, 2),
            F2HadamardSpec::plain(3, 4, 5),
            F2HadamardSpec {
                u: F2Operand::col(6),
                v: F2Operand::shifted(6, 1),
                w: F2Operand::col(7),
            },
        ];
        round_trip_batch(&columns, &specs, &[], 3);

        // Same relations, but proven + verified with the GF(2^8) scheme (the
        // embed(H₈) subspace). Validates the GF(2^8)-accelerated batch end-to-end.
        let mut tp = Blake3Transcript::new();
        let (proof, _point) = prove_oblong_and_batch_gf8(&mut tp, &columns, &specs, &[], 3);
        let mut tv = Blake3Transcript::new();
        verify_oblong_and_batch_gf8(&mut tv, &proof, &columns, &specs, &[], 3)
            .expect("GF(2^8) batched oblong round-trip + ψ_z tie");
    }

    #[test]
    fn batch_and_plus_adder_round_trips() {
        // 1 AND (U⊙V=W0, cols 0,1,2) + 1 adder t=x+y (cols 3,4,5). K=2.
        let u0 = U.to_vec();
        let v0 = V.to_vec();
        let w0: Vec<u32> = u0.iter().zip(&v0).map(|(x, y)| x & y).collect();
        let xv: Vec<u32> = V.to_vec();
        let yv: Vec<u32> = U.iter().map(|v| v.rotate_right(3)).collect();
        let tv: Vec<u32> = xv.iter().zip(&yv).map(|(a, b)| a.wrapping_add(*b)).collect();
        let columns = [
            col_from_u32s(&u0),
            col_from_u32s(&v0),
            col_from_u32s(&w0),
            col_from_u32s(&tv),
            col_from_u32s(&xv),
            col_from_u32s(&yv),
        ];
        let tm = |col, row_shift| crate::f2_hadamard::F2OperandTerm { col, row_shift };
        let and_specs = [F2HadamardSpec::plain(0, 1, 2)];
        let adder_specs = [F2AdderSpec::uniform(tm(3, 0), tm(4, 0), tm(5, 0))];
        round_trip_batch(&columns, &and_specs, &adder_specs, 3);
    }

    #[test]
    fn batch_corrupt_one_relation_is_rejected() {
        // Corrupting relation 0's W breaks its AND on some stacked rows ⇒ R₀ no
        // longer vanishes on the base domain ⇒ the eq-factored MLE-check threads
        // the wrong claim to the closing check, which rejects.
        let p: Vec<u32> = U.iter().map(|x| x.rotate_left(5) ^ 0x55).collect();
        let mut w0: Vec<u32> = U.iter().zip(&V).map(|(x, y)| x & y).collect();
        let w1: Vec<u32> = V.iter().zip(&p).map(|(x, y)| x & y).collect();
        w0[1] ^= 1 << 9; // corrupt
        let columns = [
            col_from_u32s(&U),
            col_from_u32s(&V),
            col_from_u32s(&w0),
            col_from_u32s(&V),
            col_from_u32s(&p),
            col_from_u32s(&w1),
        ];
        let specs = [F2HadamardSpec::plain(0, 1, 2), F2HadamardSpec::plain(3, 4, 5)];
        let mut tp = Blake3Transcript::new();
        let proof = prove_oblong_and_batch(&mut tp, &columns, &specs, &[], 3);
        let mut tv = Blake3Transcript::new();
        let res = verify_oblong_and_batch(&mut tv, &proof, &columns, &specs, &[], 3);
        assert!(
            matches!(res, Err(OblongVerifyError::Oblong(OblongError::FinalCheck))),
            "got {res:?}"
        );
    }
}
