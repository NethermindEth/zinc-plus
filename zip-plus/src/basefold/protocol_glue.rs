//! Protocol glue for the basefold int-lane swap: shared parameter
//! derivation (prover and verifier must agree) and (de)serialization of
//! [`Arity8Proof`] over the PCS byte stream.
//!
//! Serialization is *stream-only*: no Fiat--Shamir absorption happens here.
//! [`super::arity8::prove_batch`] and [`super::arity8::verify_batch`] absorb
//! every prover message themselves in a symmetric schedule, so the verifier
//! first deserializes the whole proof from the stream and then replays the
//! schedule against the shared `fs_transcript`.

use super::{
    BasefoldError,
    arity8::{Arity8Params, Arity8Proof, Leaf8Opening},
    chain::ChainConfigF167772161,
    chain8::{ARITY, LOG_ARITY, Radix8Chain},
    limbs::next_limb_count_arity_from_mag,
};
use crate::{
    ZipError,
    merkle::{MerkleProof, MtHash},
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};
use crypto_primitives::{PrimeField, crypto_bigint_int::Int};
use std::io::{Read, Write};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::{add, mul, sub};

/// Integer widths and digit parameters of the int-lane instantiation.
/// Digits/witness entries: `Int<BF_ND>` (192-bit; holds the 64-bit column
/// quarters and the 131-bit balanced digits). Codeword leaves: `Int<BF_NK>`
/// (384-bit; covers the radix-8 worst-case norm growth with headroom up to
/// ~6 fold rounds). Wide accumulators: `Int<BF_NW>`. Challenges:
/// `BF_NCU * 64 = 128`-bit.
pub const BF_ND: usize = 3;
pub const BF_NK: usize = 6;
pub const BF_NW: usize = 12;
pub const BF_NCU: usize = 2;
/// Balanced digit width (`lambda + LOG_ARITY + 1`).
pub const BF_P: u32 = 132;
/// Declared witness-entry magnitude exponent (`|entry| <= 2^exp`) of the
/// int lane (64-bit column quarters). Shapes the round-1 limb count and the
/// round-0 leaf serialization width.
pub const BF_INT_WITNESS_MAG_EXP: u32 = 64;
/// Declared witness-entry magnitude exponent of the binary lane (flattened
/// bits, `|entry| <= 1`).
pub const BF_BIN_WITNESS_MAG_EXP: u32 = 0;

pub fn bf_zip_err(e: BasefoldError) -> ZipError {
    ZipError::InvalidSnark(format!("basefold: {e}"))
}

/// Flatten bit-polynomial columns for the X-dimension extension: cell `b`'s
/// coefficient `l` lands at index `l + D * b` (coefficient index in the low
/// bits), so the first arity-8 fold round consumes the coefficient
/// dimension. Requires `D == ARITY`.
pub fn flatten_binary_columns<const D: usize>(
    cols: &[zinc_poly::mle::DenseMultilinearExtension<
        zinc_poly::univariate::binary::BinaryPoly<D>,
    >],
) -> Result<Vec<Vec<Int<BF_ND>>>, ZipError> {
    use num_traits::{ConstOne, ConstZero};
    if D != ARITY {
        return Err(ZipError::InvalidPcsParam(format!(
            "X-dimension extension requires cell degree bound {ARITY}, got {D}"
        )));
    }
    Ok(cols
        .iter()
        .map(|col| {
            let mut out = Vec::with_capacity(col.evaluations.len().saturating_mul(D));
            for cell in &col.evaluations {
                for b in cell.iter() {
                    let b: crypto_primitives::boolean::Boolean =
                        *std::borrow::Borrow::<crypto_primitives::boolean::Boolean>::borrow(&b);
                    out.push(if bool::from(b) {
                        Int::<BF_ND>::ONE
                    } else {
                        Int::<BF_ND>::ZERO
                    });
                }
            }
            out
        })
        .collect())
}

/// Consolidate a batch into one column (the Zip++ analogue of folded Zip):
/// interleave `cols` (each of equal power-of-two length) into a single
/// column of `2^t` times the length, `t = ceil(log2(cols.len()))`, with the
/// column index in the low `t` bits; missing columns are zero. The all-ones
/// batch claim becomes `2^t * F~(1/2,...,1/2, point)` on the result, since
/// the Lagrange weights at 1/2-coordinates are uniform (see
/// `arity8::tests::consolidation_all_ones_is_point_extension`).
pub fn interleave_columns(cols: &[Vec<Int<BF_ND>>]) -> (Vec<Int<BF_ND>>, usize) {
    use num_traits::ConstZero;
    let t = cols
        .len()
        .next_power_of_two()
        .trailing_zeros() as usize;
    let width = 1usize << t;
    let len = cols.first().map_or(0, Vec::len);
    let mut flat = vec![Int::<BF_ND>::ZERO; mul!(len, width)];
    for (j, col) in cols.iter().enumerate() {
        for (i, v) in col.iter().enumerate() {
            flat[add!(mul!(i, width), j)] = *v;
        }
    }
    (flat, t)
}

/// Convert per-poly challenge vectors into the field weight vectors of the
/// coefficient round.
pub fn alphas_to_coeff_weights<F, Chal>(
    alphas: &[Vec<Chal>],
    cfg: &F::Config,
) -> Vec<Vec<F>>
where
    F: PrimeField + for<'a> crypto_primitives::FromWithConfig<&'a Chal>,
{
    alphas
        .iter()
        .map(|a| a.iter().map(|c| F::from_with_cfg(c, cfg)).collect())
        .collect()
}

/// The arity-8 parameters of a basefold lane opening. Both sides derive
/// them from public data: the extended variable count, the lane's
/// repetition factor, its query count, and the declared witness-entry
/// magnitude exponent ([`BF_INT_WITNESS_MAG_EXP`] /
/// [`BF_BIN_WITNESS_MAG_EXP`]).
pub fn int_lane_params(
    num_vars_ext: usize,
    rep: usize,
    num_queries: usize,
    witness_mag_exp: u32,
) -> Result<Arity8Params<ChainConfigF167772161>, ZipError> {
    let msg_len = 1usize
        .checked_shl(u32::try_from(num_vars_ext).unwrap_or(u32::MAX))
        .ok_or_else(|| ZipError::InvalidPcsParam("num_vars too large".to_owned()))?;
    let full = (num_vars_ext / 3).max(1);
    // Fold-round policy from the scaling studies: fold until the tail is
    // ~2^10 entries (every extra round costs Q * k * 8 opened leaf values),
    // and build the chain two levels past the fold horizon so the tail
    // encodes through butterflies at modest value width.
    let num_rounds = num_vars_ext
        .saturating_sub(10)
        .div_ceil(3)
        .clamp(1, full);
    let depth = add!(num_rounds, 2usize).min(full);
    let chain =
        Radix8Chain::<ChainConfigF167772161>::new(msg_len, rep, depth).map_err(bf_zip_err)?;
    Ok(Arity8Params::new(chain, BF_P, num_rounds, num_queries)
        .map_err(bf_zip_err)?
        .with_witness_mag_exp(witness_mag_exp))
}

/// The limb-count schedule `k_0..=k_R` for a batch of `batch_size` columns.
/// Round 1 follows the declared witness magnitude; later rounds the digit
/// bound.
fn limb_schedule<C: super::chain::ChainConfig>(
    params: &Arity8Params<C>,
    batch_size: usize,
) -> Vec<usize> {
    let lambda = mul!(64u32, u32::try_from(BF_NCU).unwrap_or(u32::MAX));
    let mut ks = Vec::with_capacity(add!(params.num_rounds, 1usize));
    ks.push(batch_size);
    let mut k = batch_size;
    for r in 1..=params.num_rounds {
        let mag = if r == 1 {
            params.witness_mag_exp
        } else {
            sub!(params.p, 1u32)
        };
        k = next_limb_count_arity_from_mag(lambda, params.p, k, LOG_ARITY, mag);
        ks.push(k);
    }
    ks
}

/// Scratch size covering any `Int<N>` this module serializes.
const TIGHT_SCRATCH: usize = 96;

fn bytes_for_bits(bits: u32) -> usize {
    (bits as usize).div_ceil(8)
}

/// Two's-complement little-endian at an explicit byte width; the value must
/// lie in `[-2^(8*width-1), 2^(8*width-1))`.
fn write_int_tight<const N: usize>(
    transcript: &mut PcsProverTranscript,
    v: &Int<N>,
    width: usize,
) -> Result<(), ZipError> {
    let nb = <Int<N> as ConstTranscribable>::NUM_BYTES;
    debug_assert!(width > 0 && width <= nb && nb <= TIGHT_SCRATCH);
    let mut full = [0u8; TIGHT_SCRATCH];
    v.write_transcription_bytes_exact(&mut full[..nb]);
    // The dropped high bytes must be the sign fill, or the width is wrong.
    let fill = if full[sub!(width, 1usize)] & 0x80 != 0 {
        0xFF
    } else {
        0x00
    };
    if full[width..nb].iter().any(|&b| b != fill) {
        return Err(ZipError::InvalidPcsParam(
            "value exceeds its tight serialization width".to_owned(),
        ));
    }
    transcript.stream.write_all(&full[..width])?;
    Ok(())
}

fn read_int_tight<const N: usize>(
    transcript: &mut PcsVerifierTranscript,
    width: usize,
) -> Result<Int<N>, ZipError> {
    let nb = <Int<N> as ConstTranscribable>::NUM_BYTES;
    debug_assert!(width > 0 && width <= nb && nb <= TIGHT_SCRATCH);
    let mut full = [0u8; TIGHT_SCRATCH];
    transcript.stream.read_exact(&mut full[..width])?;
    let fill = if full[sub!(width, 1usize)] & 0x80 != 0 {
        0xFF
    } else {
        0x00
    };
    full[width..nb].fill(fill);
    Ok(Int::<N>::read_transcription_bytes_exact(&full[..nb]))
}

/// Byte widths of the tight proof stream, derived from public data only
/// (prover and verifier must agree): round-`r` leaf values are level-`r`
/// codeword entries of `p`-balanced-digit limbs (round 0: of the witness,
/// whose declared magnitude lives in the params), tail entries are
/// balanced digits.
pub struct TightWidths {
    pub leaf: Vec<usize>,
    pub tail: usize,
}

pub fn tight_widths<C: super::chain::ChainConfig>(params: &Arity8Params<C>) -> TightWidths {
    let leaf = (0..params.num_rounds)
        .map(|r| {
            let msg_bits = if r == 0 {
                add!(params.witness_mag_exp, 1u32)
            } else {
                params.p
            };
            let mag = params.chain.codeword_mag_bits(r, msg_bits);
            bytes_for_bits(add!(mag, 1u32)).min(<Int<BF_NK> as ConstTranscribable>::NUM_BYTES)
        })
        .collect();
    TightWidths {
        leaf,
        tail: bytes_for_bits(params.p).min(<Int<BF_ND> as ConstTranscribable>::NUM_BYTES),
    }
}

/// Write the proof into the PCS stream (no transcript absorption). Leaf
/// values, tail digits, and Merkle openings use tight widths derived from
/// the params; see [`tight_widths`].
pub fn write_arity8_proof<F, C: super::chain::ChainConfig>(
    transcript: &mut PcsProverTranscript,
    params: &Arity8Params<C>,
    proof: &Arity8Proof<F, BF_ND, BF_NK>,
) -> Result<(), ZipError>
where
    F: PrimeField,
    F::Inner: Transcribable,
{
    let widths = tight_widths(params);
    for e in &proof.initial_claims {
        transcript.write(e.inner())?;
    }
    for root in &proof.round_roots {
        transcript.write(root)?;
    }
    for round in &proof.class_claims {
        for claims in round {
            for e in claims {
                transcript.write(e.inner())?;
            }
        }
    }
    for round in &proof.limb_claims {
        for e in round {
            transcript.write(e.inner())?;
        }
    }
    for limb in &proof.tail {
        for v in limb {
            write_int_tight::<BF_ND>(transcript, v, widths.tail)?;
        }
    }
    for query in &proof.queries {
        for (r, opening) in query.iter().enumerate() {
            for v in &opening.values {
                write_int_tight::<BF_NK>(transcript, v, widths.leaf[r])?;
            }
            // Tight Merkle opening: the index in 4 bytes, then the path;
            // leaf count and path length are public (round-r tree has
            // `codeword_len_at(r + 1)` leaves).
            let idx = u32::try_from(opening.proof.leaf_index)
                .map_err(|_| ZipError::InvalidPcsParam("leaf index too large".to_owned()))?;
            transcript.stream.write_all(&idx.to_le_bytes())?;
            transcript.write_const_many(&opening.proof.siblings)?;
        }
    }
    Ok(())
}

/// Read the proof back from the PCS stream; all shapes are determined by
/// `(params, batch_size)`.
pub fn read_arity8_proof<F, C: super::chain::ChainConfig>(
    transcript: &mut PcsVerifierTranscript,
    params: &Arity8Params<C>,
    batch_size: usize,
    cfg: &F::Config,
) -> Result<Arity8Proof<F, BF_ND, BF_NK>, ZipError>
where
    F: PrimeField,
    F::Inner: Transcribable,
{
    let r_max = params.num_rounds;
    let ks = limb_schedule(params, batch_size);
    let widths = tight_widths(params);

    let read_f = |t: &mut PcsVerifierTranscript| -> Result<F, ZipError> {
        let inner: F::Inner = t.read()?;
        Ok(F::new_unchecked_with_cfg(inner, cfg))
    };

    let mut initial_claims = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        initial_claims.push(read_f(transcript)?);
    }
    let mut round_roots = Vec::with_capacity(sub!(r_max, 1usize));
    for _ in 0..sub!(r_max, 1usize) {
        round_roots.push(transcript.read::<MtHash>()?);
    }
    let mut class_claims = Vec::with_capacity(r_max);
    for r in 1..=r_max {
        let k_prev = ks[sub!(r, 1usize)];
        let mut round = Vec::with_capacity(k_prev);
        for _ in 0..k_prev {
            let mut claims = Vec::with_capacity(ARITY);
            for _ in 0..ARITY {
                claims.push(read_f(transcript)?);
            }
            let arr: [F; ARITY] = claims
                .try_into()
                .map_err(|_| ZipError::InvalidSnark("class claims".to_owned()))?;
            round.push(arr);
        }
        class_claims.push(round);
    }
    let mut limb_claims = Vec::with_capacity(sub!(r_max, 1usize));
    for &k_r in &ks[1..r_max] {
        let mut round = Vec::with_capacity(k_r);
        for _ in 0..k_r {
            round.push(read_f(transcript)?);
        }
        limb_claims.push(round);
    }
    let tail_len = params.chain.msg_len_at(r_max);
    let mut tail = Vec::with_capacity(ks[r_max]);
    for _ in 0..ks[r_max] {
        let mut limb = Vec::with_capacity(tail_len);
        for _ in 0..tail_len {
            limb.push(read_int_tight::<BF_ND>(transcript, widths.tail)?);
        }
        tail.push(limb);
    }
    let mut queries = Vec::with_capacity(params.num_queries);
    for _ in 0..params.num_queries {
        let mut openings = Vec::with_capacity(r_max);
        for (r, &k_r) in ks[..r_max].iter().enumerate() {
            let mut values = Vec::with_capacity(mul!(k_r, ARITY));
            for _ in 0..mul!(k_r, ARITY) {
                values.push(read_int_tight::<BF_NK>(transcript, widths.leaf[r])?);
            }
            let leaf_count = params.chain.codeword_len_at(add!(r, 1usize));
            let mut idx_bytes = [0u8; 4];
            transcript.stream.read_exact(&mut idx_bytes)?;
            let leaf_index = u32::from_le_bytes(idx_bytes) as usize;
            if leaf_index >= leaf_count {
                return Err(ZipError::InvalidSnark(
                    "merkle leaf index out of range".to_owned(),
                ));
            }
            let depth = leaf_count.trailing_zeros() as usize;
            let siblings = transcript.read_const_many::<MtHash>(depth)?;
            let proof = MerkleProof::new(leaf_index, leaf_count, siblings);
            openings.push(Leaf8Opening { values, proof });
        }
        queries.push(openings);
    }

    Ok(Arity8Proof {
        initial_claims,
        round_roots,
        tail,
        class_claims,
        limb_claims,
        queries,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::basefold::arity8::{commit_batch, prove_batch, verify_batch};
    use crate::pcs::structs::ZipPlusCommitment;
    use crypto_primitives::crypto_bigint_monty::MontyField;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use zinc_primality::MillerRabin;
    use zinc_transcript::traits::Transcript;

    type F = MontyField<4>;

    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn proof_stream_roundtrip() {
        // Int-lane shape: B columns of 64-bit entries, opened as their sum.
        let num_vars_ext = 11usize;
        let batch = 4usize;
        let params = int_lane_params(num_vars_ext, 4, 20, BF_INT_WITNESS_MAG_EXP).unwrap();
        let msg_len = 1usize << num_vars_ext;
        let mut rng = StdRng::seed_from_u64(11);
        let witnesses: Vec<Vec<Int<BF_ND>>> = (0..batch)
            .map(|_| {
                (0..msg_len)
                    .map(|_| {
                        let mut words = [0u64; BF_ND];
                        words[0] = rng.random();
                        Int::<BF_ND>::new(crypto_bigint::Int::from_words(words))
                    })
                    .collect()
            })
            .collect();
        let (commitment, hint) =
            commit_batch::<_, BF_ND, BF_NK, true>(&params, &witnesses).unwrap();

        // Prover side: shared fs transcript + stream.
        let mut pt = PcsProverTranscript::new_from_commitments(
            std::iter::once(&ZipPlusCommitment {
                root: commitment.root.clone(),
                batch_size: batch,
            }),
        );
        let cfg = pt
            .fs_transcript
            .get_random_field_cfg::<F, crypto_primitives::crypto_bigint_uint::Uint<4>, MillerRabin>();
        let point: Vec<F> = pt.fs_transcript.get_field_challenges(num_vars_ext, &cfg);
        let weights: Vec<F> = (0..batch).map(|_| F::one_with_cfg(&cfg)).collect();
        let (proof, eval) = prove_batch::<_, F, BF_ND, BF_NK, BF_NW, BF_NCU, true>(
            &params,
            &witnesses,
            &weights,
            None,
            &hint,
            &point,
            &cfg,
            &mut pt.fs_transcript,
        )
        .unwrap();
        write_arity8_proof(&mut pt, &params, &proof).unwrap();

        // Verifier side: fresh fs, same stream.
        let mut vt = pt.into_verification_transcript();
        vt.fs_transcript.absorb_slice(&commitment.root.0);
        let cfg_v = vt
            .fs_transcript
            .get_random_field_cfg::<F, crypto_primitives::crypto_bigint_uint::Uint<4>, MillerRabin>();
        let point_v: Vec<F> = vt.fs_transcript.get_field_challenges(num_vars_ext, &cfg_v);
        let proof_v =
            read_arity8_proof::<F, _>(&mut vt, &params, batch, &cfg_v).unwrap();
        let weights_v: Vec<F> = (0..batch).map(|_| F::one_with_cfg(&cfg_v)).collect();
        verify_batch::<_, F, BF_ND, BF_NK, BF_NW, BF_NCU, true>(
            &params,
            &commitment,
            &proof_v,
            &weights_v,
            None,
            &point_v,
            &eval,
            &cfg_v,
            &mut vt.fs_transcript,
        )
        .unwrap();
    }
}
