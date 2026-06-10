//! # Zero-knowledge masking for the Zip+ PCS ("Modular Masking")
//!
//! Implements the masked commitment and blinded opening of the companion note
//! *"Modular Masking: Zero-Knowledge IOPPs for Codes over the Rationals"*
//! (`zk_zip_plus.tex`). The construction makes the Brakedown-style opening of
//! the Zip+ PCS statistically hiding while keeping openings *exact* over the
//! integers:
//!
//! 1. **Modular Reed--Solomon masks** (paper §3, Construction 3.1). Every
//!    committed row `w_i` is encoded as `v_i = Enc(w_i) + [G s_i]_p` over `Z`,
//!    where `G` is a Reed--Solomon generator over an auxiliary prime `p`,
//!    `s_i` is a uniform seed, and `[.]_p` denotes canonical representatives
//!    in `[0, p)`. Any `<= D` opened symbols of the mask are *exactly* uniform
//!    mod `p` (Vandermonde); carries are hidden statistically because `p`
//!    exceeds the unmasked entries by `2^lambda_zk` (smudging).
//! 2. **A blinding row** `w_0`, uniform in a wide integer box, enters every
//!    revealed row combination with coefficient exactly 1, and its
//!    inner-product value `beta = <w_0, q_1>` is published *before* the
//!    combination challenge (paper §4, step 1; the simulator argument is
//!    Theorem 4.3, Hybrid 3).
//! 3. **Mask consistency**: the verifier's column check becomes a *remainder*
//!    check `rem_l = (v_0)_l + sum_j rho_j (v_j)_l - Enc(w*)_l`, which must
//!    equal the same combination of canonical mask symbols (paper Eq. (3)).
//!    In the final design this is discharged by an inner zero-knowledge IOP
//!    for the *lift relation* `R_lift` (paper §5, linear over `F_{p'}` plus
//!    range checks). **This module currently ships the transparent v0 stub**
//!    ([`mask::MaskConsistencyProof::Transparent`]): the prover reveals the
//!    seeds, the verifier recomputes the mask symbols and checks `rem`
//!    exactly. This is *binding and complete with the final round structure*,
//!    and the blinded `w*` already hides the revealed combination, but the
//!    opened columns are not yet hidden — replacing the stub with the inner
//!    ZK-IOP restores full column hiding without touching anything else.
//!
//! ## Deviations from the non-ZK base PCS (deliberate)
//!
//! - Row-combination challenges are sampled **even when `num_rows == 1`**.
//!   In the base protocol the `J = 1` challenge is vacuous; with a blinding
//!   row it is load-bearing — it separates `beta` from the claimed per-row
//!   values and gives per-row (rather than summed) proximity. See paper §6.1.
//! - Challenges here are full-width *signed* `Zt::Chal` values, so the
//!   remainder range check is two-sided: `|rem_l| <= R_max` with
//!   `R_max = (1 + sum_j |rho_j|) (p - 1)`.
//!
//! ## Scope of this increment
//!
//! Single polynomial (`batch_size = 1`), scalar evaluation lane
//! (`Zt::Comb::DEGREE_BOUND == 0`, i.e. integer evaluations; the
//! per-coefficient masking of the polynomial lanes is paper §7 item 3 and a
//! follow-up), single opening per commitment (`D = NUM_COLUMN_OPENINGS`).
//! Not yet wired into `protocol/` or the folded/multi-zip paths.
//!
//! ## Known audit items (tracked in `documentation/zk-masking-design.md`)
//!
//! - Like the base PCS, stream messages written via `write_const_many`
//!   (`w*`, column values) are not absorbed into the Fiat--Shamir state;
//!   `b` and `beta` are (via `write_field_elements`). Needs a joint audit.
//! - Knowledge soundness of the final scheme additionally relies on the
//!   affine/coset variant of mutual correlated agreement (paper
//!   Assumption 2.7).

pub mod inner;
pub mod mask;
pub mod params;
pub mod protocol;

pub use mask::{MaskConsistencyProof, MaskSeeds};
pub use params::ZkMaskParams;
pub use protocol::{ZkZip, ZkZipCommitment, ZkZipHint};
