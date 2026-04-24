//! Proof types for the logup-GKR subprotocol.

use crypto_primitives::PrimeField;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::{add, mul};

use crate::sumcheck::SumcheckProof;

/// Per-layer round proof: four "tail evaluations" of the child layer
/// at the sumcheck's output point plus the layer's sumcheck proof.
///
/// When `num_vars_parent == 0` (the layer just below the root), the
/// sumcheck is trivial and `sumcheck_proof` is `None`; the verifier
/// checks the fold identity directly from the four tail values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupGkrRoundProof<F> {
    /// `N_child(x*, 0)`.
    pub numerator_0: F,
    /// `N_child(x*, 1)`.
    pub numerator_1: F,
    /// `D_child(x*, 0)`.
    pub denominator_0: F,
    /// `D_child(x*, 1)`.
    pub denominator_1: F,
    /// Inner sumcheck proof for the layer, or `None` if the layer has
    /// 0 parent variables (first descent from the scalar root).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
}

/// Top-level logup-GKR proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupGkrProof<F> {
    /// Root numerator `N_0` (scalar, = `sum_x N_leaves(x) * prod_layers`).
    pub root_numerator: F,
    /// Root denominator `D_0` (scalar).
    pub root_denominator: F,
    /// One round proof per non-trivial layer, from top (just below the
    /// root) to bottom (just above the leaves). Length =
    /// `num_leaf_vars`.
    pub round_proofs: Vec<LogupGkrRoundProof<F>>,
}

// ---------------------------------------------------------------------------
// Transcribable impls
//
// Wire format:
//   LogupGkrRoundProof: Vec<F>(len 4: [n0, n1, d0, d1]) + u8(has_sumcheck) +
//     optional SumcheckProof (length-prefixed).
//   LogupGkrProof: Vec<F>(len 2: [root_n, root_d]) +
//     u32(num_rounds) + rounds flat (each length-prefixed as Transcribable).
// ---------------------------------------------------------------------------

impl<F: PrimeField> GenTranscribable for LogupGkrRoundProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (tails, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        assert_eq!(tails.len(), 4, "expected 4 tail values");
        let has_sc = bytes[0];
        let bytes = &bytes[1..];
        let sumcheck_proof = if has_sc == 0 {
            assert!(bytes.is_empty(), "no sumcheck proof but trailing bytes");
            None
        } else {
            let (sc, bytes) = SumcheckProof::<F>::read_transcription_bytes_subset(bytes);
            assert!(bytes.is_empty(), "trailing bytes after sumcheck proof");
            Some(sc)
        };
        let mut iter = tails.into_iter();
        let numerator_0 = iter.next().unwrap();
        let numerator_1 = iter.next().unwrap();
        let denominator_0 = iter.next().unwrap();
        let denominator_1 = iter.next().unwrap();
        Self {
            numerator_0,
            numerator_1,
            denominator_0,
            denominator_1,
            sumcheck_proof,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let tails = vec![
            self.numerator_0.clone(),
            self.numerator_1.clone(),
            self.denominator_0.clone(),
            self.denominator_1.clone(),
        ];
        let buf = tails.write_transcription_bytes_subset(buf);
        buf[0] = u8::from(self.sumcheck_proof.is_some());
        let buf = &mut buf[1..];
        if let Some(sc) = &self.sumcheck_proof {
            let buf = sc.write_transcription_bytes_subset(buf);
            assert!(buf.is_empty(), "buffer size mismatch");
        } else {
            assert!(buf.is_empty(), "buffer size mismatch");
        }
    }
}

impl<F: PrimeField> Transcribable for LogupGkrRoundProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let tails = vec![
            self.numerator_0.clone(),
            self.numerator_1.clone(),
            self.denominator_0.clone(),
            self.denominator_1.clone(),
        ];
        let tails_len = add!(Vec::<F>::LENGTH_NUM_BYTES, tails.get_num_bytes());
        let sc_len = if let Some(sc) = &self.sumcheck_proof {
            add!(SumcheckProof::<F>::LENGTH_NUM_BYTES, sc.get_num_bytes())
        } else {
            0
        };
        add!(tails_len, add!(1, sc_len))
    }
}

impl<F: PrimeField> GenTranscribable for LogupGkrProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (root_pair, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        assert_eq!(root_pair.len(), 2);
        let mut iter = root_pair.into_iter();
        let root_numerator = iter.next().unwrap();
        let root_denominator = iter.next().unwrap();

        let (num_rounds, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let num_rounds = usize::try_from(num_rounds).expect("round count fits usize");

        let mut round_proofs = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let (r, rest) = LogupGkrRoundProof::<F>::read_transcription_bytes_subset(bytes);
            round_proofs.push(r);
            bytes = rest;
        }
        assert!(bytes.is_empty(), "trailing bytes");
        Self {
            root_numerator,
            root_denominator,
            round_proofs,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let root_pair = vec![self.root_numerator.clone(), self.root_denominator.clone()];
        let buf = root_pair.write_transcription_bytes_subset(buf);

        let nrounds = u32::try_from(self.round_proofs.len()).expect("round count fits u32");
        nrounds.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        let mut buf = &mut buf[u32::NUM_BYTES..];

        for r in &self.round_proofs {
            buf = r.write_transcription_bytes_subset(buf);
        }
        assert!(buf.is_empty(), "buffer size mismatch");
    }
}

impl<F: PrimeField> Transcribable for LogupGkrProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let root_pair = vec![self.root_numerator.clone(), self.root_denominator.clone()];
        let root_len = add!(Vec::<F>::LENGTH_NUM_BYTES, root_pair.get_num_bytes());
        let rounds_len: usize = self
            .round_proofs
            .iter()
            .map(|r| add!(LogupGkrRoundProof::<F>::LENGTH_NUM_BYTES, r.get_num_bytes()))
            .sum();
        add!(root_len, add!(u32::NUM_BYTES, rounds_len))
    }
}

// No-op to silence unused `mul!` when cfg(feature = "parallel") is off.
const _: fn() = || {
    let _ = mul!(1, 1);
};
