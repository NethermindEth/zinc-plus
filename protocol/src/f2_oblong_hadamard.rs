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
//! §5, Phase D). The tie does *not* yet open the projected columns through the
//! PCS at `(z, γ)`; it checks the derived evals against the in-memory columns,
//! which validates the recombination algebra (the architectural risk). The PCS
//! opening at `γ` (vs today's `r*`) is the remaining Phase-C wiring.

use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;
use zinc_poly::univariate::oblong_and::{
    AdditiveNtt, AndCheckOutput, OblongAndProof, OblongError, WORD_BITS, base_lagrange_at,
    prove_oblong_and, verify_oblong_and,
};

use crate::f2_hadamard::{
    F2HadamardSpec, F2Operand, build_operand_column, cell_mask, derive_operand_parents,
    distinct_pairs, pair_alpha_evals,
};

type Gf = BinaryFieldGF128;

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
/// committed columns. Challenges are explicit (Fiat–Shamir is the follow-up).
pub fn prove_oblong_and_relation(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    spec: &F2HadamardSpec,
    num_vars: usize,
    r: &[Gf],
    z: Gf,
    gammas: &[Gf],
) -> OblongAndProof {
    let u = operand_words(columns, &spec.u, num_vars);
    let v = operand_words(columns, &spec.v, num_vars);
    let w = operand_words(columns, &spec.w, num_vars);
    let ntt = AdditiveNtt::new();
    prove_oblong_and(&u, &v, &w, r, z, gammas, &ntt)
}

/// Verifier: check the oblong zerocheck, then tie its operand evals to the
/// committed columns via the `ψ_z` recombination (the existing `ψ_α` machinery
/// with `base_lagrange_at(z)` weights and the row-point `γ`).
pub fn verify_oblong_and_relation(
    proof: &OblongAndProof,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    spec: &F2HadamardSpec,
    r: &[Gf],
    z: Gf,
    gammas: &[Gf],
) -> Result<(), OblongVerifyError> {
    let out: AndCheckOutput =
        verify_oblong_and(proof, r, z, gammas).map_err(OblongVerifyError::Oblong)?;

    // ψ_z recombination. eval_point = [z, γ_0, …, γ_{n-1}]; the tie evaluates at
    // the row-point γ. Feed the Lagrange weights L_i(z) where ψ_α uses α^b.
    let lagrange_z = base_lagrange_at(z).to_vec();
    let gamma = &out.eval_point[1..];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::boolean::Boolean;

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

    fn gf(seed: u64) -> Gf {
        let hi = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0x1234_5678_9ABC_DEF0;
        Gf::from_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    /// Honest oblong prove → verify (zerocheck + ψ_z tie) for the given spec.
    fn round_trip(columns: &[DenseMultilinearExtension<BinaryPoly<D>>], spec: &F2HadamardSpec, num_vars: usize) {
        let r: Vec<Gf> = (0..num_vars).map(|i| gf(i as u64 * 13 + 1)).collect();
        let z = gf(0xABCD_1234);
        let gammas: Vec<Gf> = (0..num_vars).map(|i| gf(i as u64 * 17 + 100)).collect();
        let proof = prove_oblong_and_relation(columns, spec, num_vars, &r, z, &gammas);
        verify_oblong_and_relation(&proof, columns, spec, &r, z, &gammas)
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
        // Corrupt one W word ⇒ the AND fails ⇒ oblong zerocheck rejects.
        let mut w: Vec<u32> = U.iter().zip(&V).map(|(a, b)| a & b).collect();
        w[2] ^= 1 << 7;
        let columns = [col_from_u32s(&U), col_from_u32s(&V), col_from_u32s(&w)];
        let spec = F2HadamardSpec::plain(0, 1, 2);
        let num_vars = 3;
        let r: Vec<Gf> = (0..num_vars).map(|i| gf(i as u64 + 5)).collect();
        let z = gf(0x9999);
        let gammas: Vec<Gf> = (0..num_vars).map(|i| gf(i as u64 + 200)).collect();
        let proof = prove_oblong_and_relation(&columns, &spec, num_vars, &r, z, &gammas);
        let res = verify_oblong_and_relation(&proof, &columns, &spec, &r, z, &gammas);
        assert!(
            matches!(res, Err(OblongVerifyError::Oblong(OblongError::RoundConsistency(_)))),
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
        let r: Vec<Gf> = (0..num_vars).map(|i| gf(i as u64 + 3)).collect();
        let z = gf(0x4242);
        let gammas: Vec<Gf> = (0..num_vars).map(|i| gf(i as u64 + 300)).collect();
        let proof = prove_oblong_and_relation(&columns, &F2HadamardSpec::plain(0, 1, 2), num_vars, &r, z, &gammas);
        // Verify with W tied to column 0 (= U) instead of column 2 (= W).
        let bad_spec = F2HadamardSpec::plain(0, 1, 0);
        let res = verify_oblong_and_relation(&proof, &columns, &bad_spec, &r, z, &gammas);
        assert!(matches!(res, Err(OblongVerifyError::TieMismatch)), "got {res:?}");
    }
}
