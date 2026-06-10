//! The masked commit / blinded open protocol (paper §4, Protocol 1).
//!
//! Round structure (mirroring the non-ZK Zip+ opening, plus the ZK pieces):
//!
//! ```text
//! commit : v_i = Enc(w_i) + [G s_i]_p  (i = 0 blinding row, 1..=J witness)
//!          Merkle root over masked columns + binding seed commitment
//! prove  : write b (per-row values), write beta = <w_0, q_1>      [pre-rho]
//!          sample coeffs (ALWAYS, also at num_rows == 1)
//!          write w* = w_0 + sum_j coeffs_j row_j
//!          per opening: squeeze column, write masked column + Merkle proof
//!          write mask-consistency proof (v0: seeds + salt)
//! verify : eval check <q_0, b> = eval; range check w*;
//!          coherence <w*, q_1> = beta + <coeffs, b>;
//!          per column: Merkle + leaf range + remainder range +
//!          mask-consistency (v0: recompute lifts from revealed seeds)
//! ```

use crate::{
    ZipError,
    code::LinearCode,
    merkle::{MerkleTree, MtHash},
    pcs::{
        structs::{ZipPlusParams, ZipTypes},
        utils::point_to_tensor,
        zk::{
            mask::{self, MaskSeeds},
            params::ZkMaskParams,
        },
    },
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};
use crypto_primitives::{
    FromWithConfig, PrimeField, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use num_traits::{ConstOne, ConstZero};
use rand_core::CryptoRng;
use std::marker::PhantomData;
use zinc_poly::{Polynomial, mle::DenseMultilinearExtension};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript},
};
use zinc_utils::{
    UNCHECKED, add,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul,
    mul_by_scalar::MulByScalar,
    sub,
};

/// Compact zero-knowledge commitment: the Merkle root over the *masked*
/// codeword matrix (blinding row first), plus the binding commitment to the
/// mask seeds (the stub for the inner ZK-IOP's witness oracle).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkZipCommitment {
    pub root: MtHash,
    pub seed_commitment: [u8; 32],
}

/// Prover-side opening hint for one masked commitment.
#[derive(Clone, Debug)]
pub struct ZkZipHint<Zt: ZipTypes, const WL: usize> {
    /// Masked codeword rows, `(num_rows + 1) x codeword_len`; row 0 is the
    /// blinding row.
    pub masked_rows: Vec<Vec<Int<WL>>>,
    /// The blinding row `w_0` (message side, length `row_len`).
    pub blinding_row: Vec<Zt::CombR>,
    /// Mask seeds (row 0 first). Secret until the v0 transparent opening.
    pub seeds: MaskSeeds<WL>,
    /// Salt of the seed commitment.
    pub seed_salt: [u8; 32],
    /// Merkle tree over the masked columns.
    pub merkle_tree: MerkleTree,
}

/// The zero-knowledge Zip+ PCS (single-poly, scalar evaluation lane).
pub struct ZkZip<Zt, Lc, const WL: usize>(PhantomData<(Zt, Lc)>);

impl<Zt, Lc, const WL: usize> ZkZip<Zt, Lc, WL>
where
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    Int<WL>: FromRef<Zt::Cw>
        + FromRef<Zt::CombR>
        + FromRef<Zt::Chal>
        + for<'a> MulByScalar<&'a Zt::Chal>,
    Zt::CombR: FromRef<Int<WL>>,
{
    /// Starts a prover transcript bound to the masked commitment.
    pub fn start_prover_transcript(comm: &ZkZipCommitment) -> PcsProverTranscript {
        let mut transcript =
            PcsProverTranscript::new_from_commitments(std::iter::empty::<&_>());
        Self::absorb_commitment(&mut transcript.fs_transcript, comm);
        transcript
    }

    /// Re-absorbs the commitment on the verifier side (mirrors the base PCS
    /// test pattern where the verifier re-binds the roots).
    pub fn absorb_commitment_into_verifier(
        transcript: &mut PcsVerifierTranscript,
        comm: &ZkZipCommitment,
    ) {
        Self::absorb_commitment(&mut transcript.fs_transcript, comm);
    }

    fn absorb_commitment(fs: &mut Blake3Transcript, comm: &ZkZipCommitment) {
        fs.absorb_slice(&comm.root);
        fs.absorb_slice(&comm.seed_commitment);
    }

    /// Masked commitment to a single multilinear polynomial (paper
    /// Construction 3.1). `rng` provides the *secret* prover randomness
    /// (blinding row, mask seeds, salt) — it must never be derived from the
    /// public transcript.
    pub fn commit_single(
        pp: &ZipPlusParams<Zt, Lc>,
        zkp: &ZkMaskParams<WL>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        rng: &mut impl CryptoRng,
    ) -> Result<(ZkZipHint<Zt, WL>, ZkZipCommitment), ZipError> {
        let row_len = pp.linear_code.row_len();
        let codeword_len = pp.linear_code.codeword_len();
        let num_rows = pp.num_rows;
        if poly.evaluations.len() != mul!(num_rows, row_len) {
            return Err(ZipError::InvalidPcsParam(
                "polynomial size does not match num_rows x row_len".into(),
            ));
        }
        // Each opening consumes NUM_COLUMN_OPENINGS columns of the D-wise
        // hiding budget (single-opening usage: D = C).
        if zkp.mask_dim < Zt::NUM_COLUMN_OPENINGS {
            return Err(ZipError::InvalidPcsParam(format!(
                "mask_dim ({}) below the column-opening budget ({})",
                zkp.mask_dim,
                Zt::NUM_COLUMN_OPENINGS,
            )));
        }

        // Blinding row w_0 <- [0, 2^blind_bits)^row_len, encoded wide.
        let blinding_row: Vec<Zt::CombR> = (0..row_len)
            .map(|_| {
                let sample: Int<WL> = sample_box(rng, zkp.blind_bits);
                Zt::CombR::from_ref(&sample)
            })
            .collect();
        let blinding_cw: Vec<Zt::CombR> = pp.linear_code.encode_wide(&blinding_row);

        // Unmasked codeword rows, widened to Int<WL>; blinding row first.
        let mut masked_rows: Vec<Vec<Int<WL>>> = Vec::with_capacity(add!(num_rows, 1usize));
        masked_rows.push(blinding_cw.iter().map(Int::<WL>::from_ref).collect());
        for row in poly.evaluations.chunks(row_len) {
            let cw: Vec<Zt::Cw> = pp.linear_code.encode(row);
            masked_rows.push(cw.iter().map(Int::<WL>::from_ref).collect());
        }

        // Add the modular Reed--Solomon masks over Z.
        let seeds = MaskSeeds::<WL>::sample(
            rng,
            add!(num_rows, 1usize),
            zkp.mask_dim,
            &zkp.mask_modulus,
        );
        let mask_cfg = mask::mask_field_cfg(&zkp.mask_modulus)?;
        for (row, seed) in masked_rows.iter_mut().zip(&seeds.seeds) {
            let mask_vector = mask::mask_row(seed, codeword_len, &mask_cfg);
            for (entry, symbol) in row.iter_mut().zip(&mask_vector) {
                *entry = add!(
                    *entry,
                    &uint_to_int::<WL>(symbol),
                    "masked codeword entry overflow"
                );
            }
        }

        let row_refs: Vec<&[Int<WL>]> = masked_rows.iter().map(Vec::as_slice).collect();
        let merkle_tree = MerkleTree::new(&row_refs);
        let root = merkle_tree.root();

        let mut seed_salt = [0u8; 32];
        rng.fill_bytes(&mut seed_salt);
        let seed_commitment = seeds.commitment(&seed_salt);

        Ok((
            ZkZipHint {
                masked_rows,
                blinding_row,
                seeds,
                seed_salt,
                merkle_tree,
            },
            ZkZipCommitment {
                root,
                seed_commitment,
            },
        ))
    }

    /// Blinded opening at `point_f` (paper §4, prover side). Returns the
    /// evaluation `<q_0, b>` over `F`.
    #[allow(clippy::arithmetic_side_effects)] // field ops + bounded index arithmetic
    pub fn prove_single<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        zkp: &ZkMaskParams<WL>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        point_f: &[F],
        hint: &ZkZipHint<Zt, WL>,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();
        if Zt::Comb::DEGREE_BOUND != 0 {
            return Err(ZipError::InvalidPcsParam(
                "zk opening currently supports the scalar lane only \
                 (Comb::DEGREE_BOUND == 0); polynomial lanes need \
                 per-coefficient masks (paper §7, item 3)"
                    .into(),
            ));
        }
        // One opening consumes the full column budget of the mask dimension.
        if zkp.mask_dim < Zt::NUM_COLUMN_OPENINGS {
            return Err(ZipError::InvalidPcsParam(
                "mask_dim below the column-opening budget".into(),
            ));
        }
        if point_f.len() != pp.num_vars {
            return Err(ZipError::InvalidPcsParam(
                "evaluation point has wrong length".into(),
            ));
        }

        let (q_0, q_1) = point_to_tensor(num_rows, point_f, field_cfg)?;
        let zero_f = F::zero_with_cfg(field_cfg);

        // Scalar lane: alpha-projection is trivial (alphas = [1]).
        let alphas = vec![Zt::Chal::ONE];
        let poly_comb_r: Vec<Zt::CombR> = poly
            .evaluations
            .iter()
            .map(|eval| {
                Zt::EvalDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                    eval,
                    &alphas,
                    Zt::CombR::ZERO,
                )
                .map_err(ZipError::from)
            })
            .collect::<Result<_, _>>()?;

        // Per-row claimed values b (witness rows only) — absorbed into FS.
        let b: Vec<F> = poly_comb_r
            .chunks(row_len)
            .map(|row| MBSInnerProduct::inner_product_field(row, &q_1, zero_f.clone()))
            .collect::<Result<_, _>>()?;
        transcript.write_field_elements(&b)?;

        // beta = <w_0, q_1>: the blinding row's claimed value, published
        // before the combination challenge (load-bearing: Thm 4.2 / 4.3).
        let beta: F =
            MBSInnerProduct::inner_product_field(&hint.blinding_row, &q_1, zero_f.clone())?;
        transcript.write_field_elements(std::slice::from_ref(&beta))?;

        let eval = MBSInnerProduct::inner_product::<UNCHECKED>(&q_0, &b, zero_f.clone())?;

        // ZK deviation from the base PCS: challenges are sampled even when
        // num_rows == 1 (the J = 1 inversion, paper §6.1).
        let coeffs: Vec<Zt::Chal> = transcript.fs_transcript.get_challenges(num_rows);

        // w* = w_0 + sum_j coeffs_j row_j over CombR.
        let mut w_star: Vec<Zt::CombR> = hint.blinding_row.clone();
        for (row, coeff) in poly_comb_r.chunks(row_len).zip(&coeffs) {
            for (acc, entry) in w_star.iter_mut().zip(row) {
                let scaled: Zt::CombR = entry
                    .mul_by_scalar::<CHECK_FOR_OVERFLOW>(coeff)
                    .ok_or_else(|| ZipError::InvalidPcsOpen("w* scaling overflow".into()))?;
                *acc = add!(*acc, &scaled, "w* accumulation overflow");
            }
        }
        transcript.write_const_many(&w_star)?;
        // Unlike the base PCS, bind every opening message into the FS state
        // (column challenges must depend on w*, and each on the previous
        // openings).
        absorb_const_many(&mut transcript.fs_transcript, &w_star);

        // Column openings over the masked matrix.
        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            let column_values: Vec<Int<WL>> = hint
                .masked_rows
                .iter()
                .map(|row| row[column_idx])
                .collect();
            transcript.write_const_many(&column_values)?;
            absorb_const_many(&mut transcript.fs_transcript, &column_values);
            let merkle_proof = hint
                .merkle_tree
                .prove(column_idx)
                .map_err(|_| ZipError::InvalidPcsOpen("Failed to open merkle tree".into()))?;
            transcript.write_merkle_proof(&merkle_proof)?;
        }

        // v0 transparent mask-consistency opening: reveal seeds + salt.
        // TODO(zk-inner): replace with the R_lift ZK-IOP (paper §5); this is
        // the only step that breaks column hiding.
        for seed in &hint.seeds.seeds {
            transcript.write_const_many(seed)?;
            absorb_const_many(&mut transcript.fs_transcript, seed);
        }
        transcript.write(&MtHash::from(hint.seed_salt))?;
        transcript.fs_transcript.absorb_slice(&hint.seed_salt);

        Ok(eval)
    }

    /// Verifies a blinded opening (paper §4, verifier side).
    #[allow(clippy::arithmetic_side_effects)] // field ops + bounded index arithmetic
    pub fn verify<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vp: &ZipPlusParams<Zt, Lc>,
        zkp: &ZkMaskParams<WL>,
        comm: &ZkZipCommitment,
        field_cfg: &F::Config,
        point_f: &[F],
        eval_f: &F,
    ) -> Result<(), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let num_rows = vp.num_rows;
        let row_len = vp.linear_code.row_len();
        let codeword_len = vp.linear_code.codeword_len();
        if point_f.len() != vp.num_vars {
            return Err(ZipError::InvalidPcsParam(
                "evaluation point has wrong length".into(),
            ));
        }

        let (q_0, q_1) = point_to_tensor(num_rows, point_f, field_cfg)?;
        let zero_f = F::zero_with_cfg(field_cfg);

        // Check 1: <q_0, b> == eval.
        let b: Vec<F> = transcript.read_field_elements(num_rows)?;
        if MBSInnerProduct::inner_product::<UNCHECKED>(&q_0, &b, zero_f.clone())? != *eval_f {
            return Err(ZipError::InvalidPcsOpen(
                "Evaluation consistency failure".into(),
            ));
        }

        let beta: F = transcript
            .read_field_elements::<F>(1)?
            .pop()
            .ok_or_else(|| ZipError::InvalidPcsOpen("missing beta".into()))?;

        let coeffs: Vec<Zt::Chal> = transcript.fs_transcript.get_challenges(num_rows);

        // Check 2: w* range (|w*|_inf < 2^(blind_bits + 1)). Part of the soft
        // bit-size enforcement; with blinding the bound is mask-dominated and
        // witness-independent.
        let w_star: Vec<Zt::CombR> = transcript.read_const_many(row_len)?;
        absorb_const_many(&mut transcript.fs_transcript, &w_star);
        for entry in &w_star {
            if Int::<WL>::from_ref(entry).inner().abs().bits() > add!(zkp.blind_bits, 1u32) {
                return Err(ZipError::InvalidPcsOpen("w* entry out of range".into()));
            }
        }

        // Check 3 (ZK coherence): <w*, q_1> == beta + <coeffs, b>.
        let lhs = MBSInnerProduct::inner_product_field(&w_star, &q_1, zero_f.clone())?;
        let rhs = beta + MBSInnerProduct::inner_product_field(&coeffs, &b, zero_f.clone())?;
        if lhs != rhs {
            return Err(ZipError::InvalidPcsOpen("Coherence failure (zk)".into()));
        }

        let encoded_w_star: Vec<Int<WL>> = vp
            .linear_code
            .encode_wide(&w_star)
            .iter()
            .map(Int::<WL>::from_ref)
            .collect();

        // R_max = (1 + sum_j |coeffs_j|) * (p - 1); remainders are two-sided
        // because challenges are signed.
        let one = Uint::<WL>::from_u64(1);
        let p_minus_1 = sub!(zkp.mask_modulus, &one, "mask modulus is zero");
        let mut weight_sum = one;
        for coeff in &coeffs {
            weight_sum = add!(
                weight_sum,
                &Uint::new(Int::<WL>::from_ref(coeff).inner().abs()),
                "R_max weight overflow"
            );
        }
        let r_max = mul!(weight_sum, &p_minus_1, "R_max overflow");

        // Committed leaves are mask + bounded encoding: |v| < 2p.
        let leaf_bits = add!(zkp.mask_modulus_bits, 1u32);

        // Checks 4-6 per opened column: Merkle, leaf range, remainder range.
        let mut opened: Vec<(usize, Vec<Int<WL>>)> =
            Vec::with_capacity(Zt::NUM_COLUMN_OPENINGS);
        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(codeword_len);
            let column_values: Vec<Int<WL>> =
                transcript.read_const_many(add!(num_rows, 1usize))?;
            absorb_const_many(&mut transcript.fs_transcript, &column_values);
            let merkle_proof = transcript.read_merkle_proof()?;
            merkle_proof
                .verify(&comm.root, &column_values, column_idx)
                .map_err(|e| {
                    ZipError::InvalidPcsOpen(format!("Column opening verification failed: {e}"))
                })?;
            if column_values.iter().any(|v| v.inner().abs().bits() > leaf_bits) {
                return Err(ZipError::InvalidPcsOpen(
                    "committed column entry out of range".into(),
                ));
            }
            opened.push((column_idx, column_values));
        }

        // v0 transparent mask-consistency proof: read seeds, re-bind to the
        // seed commitment, recompute the canonical lifts.
        // TODO(zk-inner): replace with the R_lift ZK-IOP verifier (paper §5).
        let seeds = MaskSeeds::<WL> {
            seeds: (0..=num_rows)
                .map(|_| {
                    let seed = transcript.read_const_many::<Uint<WL>>(zkp.mask_dim)?;
                    absorb_const_many(&mut transcript.fs_transcript, &seed);
                    Ok::<_, ZipError>(seed)
                })
                .collect::<Result<_, _>>()?,
        };
        let salt: MtHash = transcript.read()?;
        transcript.fs_transcript.absorb_slice(&salt);
        let salt_bytes: [u8; 32] = salt
            .as_ref()
            .try_into()
            .map_err(|_| ZipError::InvalidPcsOpen("malformed seed salt".into()))?;
        if seeds.commitment(&salt_bytes) != comm.seed_commitment {
            return Err(ZipError::InvalidPcsOpen(
                "mask seed commitment mismatch".into(),
            ));
        }
        // Canonical-residue checks are load-bearing (paper Thm 4.2): without
        // them the effective mask is not fixed before the challenge.
        seeds.validate(zkp.mask_dim, &zkp.mask_modulus)?;

        let mask_cfg = mask::mask_field_cfg(&zkp.mask_modulus)?;
        for (column_idx, column_values) in &opened {
            // rem = v_0 + sum_j coeffs_j v_j - Enc(w*)[l].
            let mut rem = column_values[0];
            for (value, coeff) in column_values[1..].iter().zip(&coeffs) {
                let scaled: Int<WL> = value
                    .mul_by_scalar::<CHECK_FOR_OVERFLOW>(coeff)
                    .ok_or_else(|| ZipError::InvalidPcsOpen("rem scaling overflow".into()))?;
                rem = add!(rem, &scaled, "rem accumulation overflow");
            }
            rem = sub!(rem, &encoded_w_star[*column_idx], "rem subtraction overflow");

            // Remainder range check (paper §4 step 5, two-sided).
            if Uint::new(rem.inner().abs()) > r_max {
                return Err(ZipError::InvalidPcsOpen(
                    "remainder out of range".into(),
                ));
            }

            // Lift check: rem == X_0 + sum_j coeffs_j X_j with canonical
            // lifts X_i = [<G_l, s_i>]_p (paper Eq. (3)).
            let mut expected =
                uint_to_int::<WL>(&mask::mask_symbol_at(&seeds.seeds[0], *column_idx, &mask_cfg));
            for (seed, coeff) in seeds.seeds[1..].iter().zip(&coeffs) {
                let lift =
                    uint_to_int::<WL>(&mask::mask_symbol_at(seed, *column_idx, &mask_cfg));
                let scaled: Int<WL> = lift
                    .mul_by_scalar::<CHECK_FOR_OVERFLOW>(coeff)
                    .ok_or_else(|| ZipError::InvalidPcsOpen("lift scaling overflow".into()))?;
                expected = add!(expected, &scaled, "lift accumulation overflow");
            }
            if rem != expected {
                return Err(ZipError::InvalidPcsOpen(
                    "mask consistency failure".into(),
                ));
            }
        }

        Ok(())
    }
}

/// Binds a slice of fixed-width transcript values into the Fiat--Shamir
/// state (the byte-stream writes alone do not touch it).
fn absorb_const_many<T: ConstTranscribable>(fs: &mut Blake3Transcript, values: &[T]) {
    let mut buf = vec![0u8; T::NUM_BYTES];
    for value in values {
        value.write_transcription_bytes_exact(&mut buf);
        fs.absorb_slice(&buf);
    }
}

/// Reinterprets a (small enough) `Uint` as a non-negative `Int` (two's
/// complement reinterpretation; the parameter derivation guarantees the sign
/// bit is clear).
fn uint_to_int<const WL: usize>(value: &Uint<WL>) -> Int<WL> {
    debug_assert!(
        value.inner().bits() < u32::try_from(mul!(WL, 64usize)).expect("WL fits u32"),
        "uint_to_int requires headroom for the sign bit"
    );
    *value.as_int()
}

/// Uniform sample from the box `[0, 2^bits)` as a non-negative `Int<WL>`.
#[allow(clippy::arithmetic_side_effects)] // byte/bit index arithmetic bounded by construction
fn sample_box<const WL: usize>(rng: &mut impl CryptoRng, bits: u32) -> Int<WL> {
    debug_assert!(usize::try_from(bits).expect("bits fits usize") < WL * 64 - 1);
    let full_bytes = usize::try_from(bits.div_ceil(8)).expect("bit count fits usize");
    let top_mask: u8 = match bits % 8 {
        0 => 0xff,
        r => u8::MAX >> (8 - r),
    };
    let mut buf = vec![0u8; mul!(WL, 8usize)];
    rng.fill_bytes(&mut buf[..full_bytes]);
    if !bits.is_multiple_of(8)
        && let Some(last) = buf[..full_bytes].last_mut()
    {
        *last &= top_mask;
    }
    let value = Uint::new(crypto_bigint::Uint::from_le_slice(&buf));
    uint_to_int(&value)
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::{
        code::iprs::IprsCode,
        pcs::{
            structs::ZipPlus,
            test_utils::{IPRS_DEPTH, IPRS_ROW_LEN, REP_FACTOR, TestIprsConfig, TestZipTypes},
        },
    };
    use crypto_primitives::{IntoWithConfig, crypto_bigint_monty::MontyField};
    use rand::{SeedableRng, rngs::StdRng};
    use zinc_utils::CHECKED;

    type Zt = TestZipTypes<1, 2, 4>;
    type Lc = IprsCode<Zt, TestIprsConfig, REP_FACTOR, CHECKED>;
    type F = MontyField<2>;
    const WL: usize = 8;
    type Zk = ZkZip<Zt, Lc, WL>;

    const LAMBDA_ZK: u32 = 64;

    type Setup = (
        ZipPlusParams<Zt, Lc>,
        ZkMaskParams<WL>,
        DenseMultilinearExtension<<Zt as ZipTypes>::Eval>,
    );

    fn setup(num_vars: usize) -> Setup {
        let code = IprsCode::new(IPRS_ROW_LEN, IPRS_DEPTH).expect("iprs code");
        let pp = ZipPlus::<Zt, Lc>::setup(1 << num_vars, code);
        let zkp = ZkMaskParams::<WL>::derive_for_code(
            &pp,
            LAMBDA_ZK,
            <Zt as ZipTypes>::NUM_COLUMN_OPENINGS,
        )
        .expect("zk params");
        let evaluations = (1..=(1i32 << num_vars)).map(Int::from).collect();
        let poly = DenseMultilinearExtension {
            num_vars,
            evaluations,
        };
        (pp, zkp, poly)
    }

    #[allow(clippy::type_complexity)]
    fn prove_roundtrip(
        num_vars: usize,
        rng_seed: u64,
    ) -> (
        Setup,
        ZkZipCommitment,
        Vec<F>,
        F,
        PcsVerifierTranscript,
        <F as PrimeField>::Config,
    ) {
        let (pp, zkp, poly) = setup(num_vars);
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let (hint, comm) = Zk::commit_single(&pp, &zkp, &poly, &mut rng).expect("commit");

        let mut transcript = Zk::start_prover_transcript(&comm);
        let field_cfg = transcript
            .fs_transcript
            .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>();
        let point: Vec<<Zt as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();
        let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let eval = Zk::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &zkp,
            &poly,
            &point_f,
            &hint,
            &field_cfg,
        )
        .expect("prove");

        let mut transcript = transcript.into_verification_transcript();
        Zk::absorb_commitment_into_verifier(&mut transcript, &comm);
        let verifier_cfg = transcript
            .fs_transcript
            .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>();

        ((pp, zkp, poly), comm, point_f, eval, transcript, verifier_cfg)
    }

    #[test]
    fn roundtrip_multi_row() {
        // num_vars = 10 with row_len 256 gives num_rows = 4.
        let ((pp, zkp, _), comm, point_f, eval, mut transcript, cfg) = prove_roundtrip(10, 1);
        Zk::verify::<F, CHECKED>(&mut transcript, &pp, &zkp, &comm, &cfg, &point_f, &eval)
            .expect("verification must pass");
    }

    #[test]
    fn roundtrip_single_row() {
        // num_vars = 8 = log2(row_len): the J = 1 regime (paper §6.1); the
        // combination challenge must still be sampled.
        let ((pp, zkp, _), comm, point_f, eval, mut transcript, cfg) = prove_roundtrip(8, 2);
        assert_eq!(pp.num_rows, 1);
        Zk::verify::<F, CHECKED>(&mut transcript, &pp, &zkp, &comm, &cfg, &point_f, &eval)
            .expect("verification must pass at J = 1");
    }

    #[test]
    fn wrong_eval_is_rejected() {
        let ((pp, zkp, _), comm, point_f, eval, mut transcript, cfg) = prove_roundtrip(10, 3);
        let wrong = eval + F::one_with_cfg(&cfg);
        let err = Zk::verify::<F, CHECKED>(&mut transcript, &pp, &zkp, &comm, &cfg, &point_f, &wrong)
            .expect_err("wrong evaluation must be rejected");
        assert!(matches!(err, ZipError::InvalidPcsOpen(_)));
    }

    #[test]
    fn tampered_seed_commitment_is_rejected() {
        let ((pp, zkp, _), comm, point_f, eval, mut transcript, cfg) = prove_roundtrip(10, 4);
        let mut tampered = comm;
        tampered.seed_commitment[0] ^= 1;
        let err = Zk::verify::<F, CHECKED>(
            &mut transcript,
            &pp,
            &zkp,
            &tampered,
            &cfg,
            &point_f,
            &eval,
        )
        .expect_err("tampered seed commitment must be rejected");
        assert!(matches!(err, ZipError::InvalidPcsOpen(_)));
    }

    #[test]
    fn commitment_is_randomized() {
        // Hiding smoke test: same polynomial, different prover randomness
        // => different roots and different seed commitments.
        let (pp, zkp, poly) = setup(10);
        let mut rng_a = StdRng::seed_from_u64(10);
        let mut rng_b = StdRng::seed_from_u64(11);
        let (_, comm_a) = Zk::commit_single(&pp, &zkp, &poly, &mut rng_a).expect("commit a");
        let (_, comm_b) = Zk::commit_single(&pp, &zkp, &poly, &mut rng_b).expect("commit b");
        assert_ne!(comm_a.root, comm_b.root);
        assert_ne!(comm_a.seed_commitment, comm_b.seed_commitment);
    }

    #[test]
    fn insufficient_mask_budget_is_rejected() {
        let (pp, mut zkp, poly) = setup(10);
        zkp.mask_dim = <Zt as ZipTypes>::NUM_COLUMN_OPENINGS - 1;
        let mut rng = StdRng::seed_from_u64(12);
        assert!(Zk::commit_single(&pp, &zkp, &poly, &mut rng).is_err());
    }
}
