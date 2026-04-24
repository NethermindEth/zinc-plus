//! Leaf-MLE construction for a single lookup group.
//!
//! Bridges between the abstract logup-GKR subprotocol (which proves
//! `sum_x N(x) / D(x) = S` for arbitrary `N, D`) and the concrete
//! lookup identity:
//!
//! ```text
//! sum_{l, i} -1 / (alpha - c_l[i]) + sum_j m[j] / (alpha - T[j]) = 0
//! ```
//!
//! Layout of the flat leaf index `b`:
//!
//! * Low `row_vars` bits: the row index `i` (or `j` for the table slot).
//! * High `slot_vars` bits: the slot index, mapped as
//!   - `slot in 0..L` -> witness column `l = slot`
//!   - `slot = L`     -> the table slot (numerator carries multiplicity)
//!   - `slot > L`     -> padding (numerator 0, denominator 1 so n/d = 0)
//!
//! `slot_vars = ceil(log2(L + 1))`; `row_vars = log2(row_count)`.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::{add, inner_transparent_field::InnerTransparentField};

/// Output of [`build_lookup_leaves`]: the leaf MLEs plus shape metadata
/// that the verifier uses to split the final GKR point.
#[derive(Clone, Debug)]
pub struct LookupLeaves<F: PrimeField> {
    pub numerator: DenseMultilinearExtension<F::Inner>,
    pub denominator: DenseMultilinearExtension<F::Inner>,
    /// Number of witness columns bundled into this group (L).
    pub num_witness_columns: usize,
    /// Number of row variables.
    pub row_vars: usize,
    /// Number of slot variables (= `ceil(log2(L + 1))`).
    pub slot_vars: usize,
}

impl<F: PrimeField> LookupLeaves<F> {
    /// Total leaf variables = `row_vars + slot_vars`.
    pub fn total_vars(&self) -> usize {
        self.row_vars + self.slot_vars
    }
}

/// Build the leaf `(N, D)` MLEs for a single lookup group.
///
/// All witness columns, the table, and the multiplicity vector must
/// share the same number of rows `2^row_vars` (the caller is
/// responsible for padding the table / multiplicities if shorter).
///
/// # Panics
///
/// * If `witness_columns` is empty.
/// * If any MLE's `num_vars` disagrees with the others.
#[allow(clippy::arithmetic_side_effects)]
pub fn build_lookup_leaves<F>(
    witness_columns: &[&DenseMultilinearExtension<F::Inner>],
    table: &DenseMultilinearExtension<F::Inner>,
    multiplicities: &DenseMultilinearExtension<F::Inner>,
    alpha: &F,
    cfg: &F::Config,
) -> LookupLeaves<F>
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig,
{
    assert!(!witness_columns.is_empty(), "need at least one witness column");
    let row_vars = witness_columns[0].num_vars;
    for col in witness_columns {
        assert_eq!(col.num_vars, row_vars, "all witness columns must share num_vars");
    }
    assert_eq!(table.num_vars, row_vars, "table num_vars must equal row_vars");
    assert_eq!(
        multiplicities.num_vars, row_vars,
        "multiplicities num_vars must equal row_vars"
    );

    let n_witness = witness_columns.len();
    let n_slots = n_witness + 1; // +1 for the table slot
    let slot_vars = n_slots.next_power_of_two().trailing_zeros() as usize;
    let padded_slots = 1usize << slot_vars;
    let row_count = 1usize << row_vars;
    let total_len = padded_slots * row_count;

    let zero_inner = F::zero_with_cfg(cfg).into_inner();
    let one_inner = F::from_with_cfg(1u64, cfg).into_inner();

    let mut n_evals: Vec<F::Inner> = vec![zero_inner.clone(); total_len];
    let mut d_evals: Vec<F::Inner> = vec![one_inner.clone(); total_len];

    // Witness slots: N = -1, D = alpha - c_l[i].
    let minus_one = -F::from_with_cfg(1u64, cfg);
    let minus_one_inner = minus_one.into_inner();
    for (l, col) in witness_columns.iter().enumerate() {
        let base = l * row_count;
        for i in 0..row_count {
            n_evals[base + i] = minus_one_inner.clone();
            let c_val = F::new_unchecked_with_cfg(col.evaluations[i].clone(), cfg);
            let d_val = alpha.clone() - &c_val;
            d_evals[base + i] = d_val.into_inner();
        }
    }

    // Table slot (slot = L): N = m[j], D = alpha - T[j].
    let table_base = n_witness * row_count;
    for j in 0..row_count {
        let m_val = F::new_unchecked_with_cfg(multiplicities.evaluations[j].clone(), cfg);
        n_evals[table_base + j] = m_val.into_inner();
        let t_val = F::new_unchecked_with_cfg(table.evaluations[j].clone(), cfg);
        let d_val = alpha.clone() - &t_val;
        d_evals[table_base + j] = d_val.into_inner();
    }

    // Padding slots (slots L+1..padded_slots) have N=0 (already), D=1 (already).

    let total_vars = row_vars + slot_vars;
    LookupLeaves {
        numerator: DenseMultilinearExtension::from_evaluations_vec(
            total_vars,
            n_evals,
            zero_inner.clone(),
        ),
        denominator: DenseMultilinearExtension::from_evaluations_vec(
            total_vars,
            d_evals,
            zero_inner,
        ),
        num_witness_columns: n_witness,
        row_vars,
        slot_vars,
    }
}

/// Component evaluations at the row-part of the final GKR point.
///
/// The caller — who owns the actual trace commitments — supplies
/// these. The verifier combines them with the slot-part Lagrange
/// values to reconstruct `N(rho)` and `D(rho)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafComponentEvals<F> {
    /// `c_l(rho_row)` for each witness column `l`.
    pub witness_evals: Vec<F>,
    /// `T(rho_row)`.
    pub table_eval: F,
    /// `m(rho_row)`.
    pub multiplicity_eval: F,
}

/// Reconstruct the expected `(N(rho), D(rho))` from component evals
/// and the slot-part of the final point.
///
/// The identity:
///
/// ```text
/// N(rho) = sum_l  E_l * (-1)  +  E_T * m(rho_row)
/// D(rho) = sum_l  E_l * (alpha - c_l(rho_row))
///        + E_T * (alpha - T(rho_row))
///        + E_pad * 1
/// ```
///
/// where `E_l = eq(rho_slot, l_binary)`, `E_T = eq(rho_slot, L_binary)`,
/// and `E_pad = 1 - sum_l E_l - E_T` (the remaining Lagrange mass on
/// the padded slots).
#[allow(clippy::arithmetic_side_effects)]
pub fn expected_leaf_evals<F>(
    rho_slot: &[F],
    components: &LeafComponentEvals<F>,
    alpha: &F,
    cfg: &F::Config,
) -> (F, F)
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig,
{
    let one = F::from_with_cfg(1u64, cfg);
    let num_witness = components.witness_evals.len();
    let num_slots = num_witness + 1;

    // Build Lagrange basis values for the slot point.
    //
    // lagrange[i] = eq(rho_slot, i_binary) for i in 0..2^slot_vars.
    let lagrange = lagrange_basis::<F>(rho_slot, cfg);
    assert!(
        lagrange.len() >= num_slots,
        "slot point dim does not fit L+1 slots"
    );

    // Sum of E_l for l in 0..L.
    let mut sum_witness_e = F::zero_with_cfg(cfg);
    // N(rho) = -sum_witness_e + E_T * m_eval
    // D(rho) starts at 0 and accumulates.
    let mut n_rho = F::zero_with_cfg(cfg);
    let mut d_rho = F::zero_with_cfg(cfg);

    for (l, c_eval) in components.witness_evals.iter().enumerate() {
        let e_l = &lagrange[l];
        sum_witness_e = sum_witness_e.clone() + e_l;
        // N contribution: -E_l.
        n_rho = n_rho - e_l;
        // D contribution: E_l * (alpha - c_l).
        let diff = alpha.clone() - c_eval;
        d_rho = d_rho.clone() + e_l.clone() * &diff;
    }

    let e_t = &lagrange[num_witness];
    // N contribution: E_T * m_eval.
    n_rho = n_rho + e_t.clone() * &components.multiplicity_eval;
    // D contribution: E_T * (alpha - T_eval).
    let diff_t = alpha.clone() - &components.table_eval;
    d_rho = d_rho + e_t.clone() * &diff_t;

    // E_pad = 1 - sum_witness_e - E_T, and it multiplies the padding
    // denominator value 1.
    let e_pad = one - &sum_witness_e - e_t;
    d_rho = d_rho + e_pad;

    (n_rho, d_rho)
}

/// Precompute the Lagrange basis `eq(r, i_binary)` for `i` in
/// `0..2^r.len()`, consistent with the MLE indexing convention used by
/// `build_eq_x_r_vec` (LSB of `i` = `r[0]`).
#[allow(clippy::arithmetic_side_effects)]
fn lagrange_basis<F>(r: &[F], cfg: &F::Config) -> Vec<F>
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig,
{
    let one = F::from_with_cfg(1u64, cfg);
    if r.is_empty() {
        return vec![one];
    }
    let mut buf: Vec<F> = vec![one.clone() - &r[0], r[0].clone()];
    for ri in &r[1..] {
        let mut next = Vec::with_capacity(buf.len() * 2);
        let one_minus_ri = one.clone() - ri;
        for v in &buf {
            next.push(v.clone() * &one_minus_ri);
        }
        for v in &buf {
            next.push(v.clone() * ri);
        }
        buf = next;
    }
    buf
}

// ---------------------------------------------------------------------------
// Transcribable for LeafComponentEvals
//
// Wire format: Vec<F>(witness_evals) + Vec<F>(len 2: [table_eval, mul_eval]).
// ---------------------------------------------------------------------------

impl<F: PrimeField> GenTranscribable for LeafComponentEvals<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (witness_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        let (tm_pair, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        assert!(bytes.is_empty(), "trailing bytes");
        assert_eq!(tm_pair.len(), 2, "expected table+mul pair");
        let mut it = tm_pair.into_iter();
        let table_eval = it.next().unwrap();
        let multiplicity_eval = it.next().unwrap();
        Self {
            witness_evals,
            table_eval,
            multiplicity_eval,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.witness_evals.write_transcription_bytes_subset(buf);
        let pair = vec![self.table_eval.clone(), self.multiplicity_eval.clone()];
        let buf = pair.write_transcription_bytes_subset(buf);
        assert!(buf.is_empty(), "buffer size mismatch");
    }
}

impl<F: PrimeField> Transcribable for LeafComponentEvals<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let pair = vec![self.table_eval.clone(), self.multiplicity_eval.clone()];
        add!(
            Vec::<F>::LENGTH_NUM_BYTES,
            add!(
                self.witness_evals.get_num_bytes(),
                add!(Vec::<F>::LENGTH_NUM_BYTES, pair.get_num_bytes())
            )
        )
    }
}
