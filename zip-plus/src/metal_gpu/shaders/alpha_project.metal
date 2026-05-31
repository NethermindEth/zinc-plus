// =====================================================================
// alpha_project.metal — column-wise α-projection of F_2 cells to GF(2^128).
//
// CPU mirror: `eval_f2_poly_d_at_with_powers` /
// `eval_f2_poly_d_at_with_powers_simd_x4` /
// `project_column_with_powers` in
// `poly/src/univariate/binary_gf128.rs`.
//
// Layout:
//   * `cells`        — flat `ulong[]`, length `num_cells`.
//                      Each `ulong` holds one `BinaryPoly<D>` cell;
//                      only bits 0..D carry data (D <= 32 in the
//                      f2-clean SHA-256 pipeline, but the kernel
//                      tolerates D <= 64).
//                      Byte-equivalent to the CPU side's
//                      `&[BinaryPoly<D>]` with the `simd` feature
//                      enabled (BinaryU64Poly = #[repr(transparent)] u64).
//   * `alpha_pows`   — flat `ulong2[]`, length >= D. Powers
//                      α^0, α^1, ..., α^{D-1} of the projection
//                      challenge in GF(2^128). `ulong2 = [u64; 2]`,
//                      matching `BinaryFieldGF128::words()`.
//   * `out`          — flat `ulong2[]`, length `num_cells`. Per-cell
//                      F_2-poly evaluation at α, in GF(2^128).
//   * `params.num_cells` — number of cells in the column.
//   * `params.d`         — number of bits per cell (== D template
//                      arg on the CPU side). `d <= 64`.
//
// Thread grid: launch `num_cells` threads. Thread `gid` projects
// cell `gid`.
//
// Operation:
//   acc = 0
//   for i in 0..D:
//       if (cell >> i) & 1: acc ^= alpha_pows[i]
//   out[gid] = acc
//
// This is the bolt-rs `expander_inner_product.metal` template
// stripped of clmul + modular reduction — for the α-projection,
// the F_2 coefficients are 0/1 so multiplication by them is just
// masking, and accumulation is XOR (= GF(2^k) addition).
// =====================================================================

#include <metal_stdlib>
using namespace metal;

struct AlphaProjectParams {
    uint num_cells;
    uint d;
};

kernel void project_columns_at_alpha(
    device const ulong*  cells      [[buffer(0)]],
    device const ulong2* alpha_pows [[buffer(1)]],
    device       ulong2* out        [[buffer(2)]],
    constant AlphaProjectParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.num_cells) {
        return;
    }

    ulong cell = cells[gid];
    ulong2 acc = ulong2(0, 0);

    // Branchless masked-XOR. Each iteration: read one α-power (16 B,
    // amortised across the threadgroup since the same i is touched
    // by every thread at the same step) and XOR it conditionally
    // into the accumulator based on bit i of the cell. No clmul, no
    // reduction — just `mask & alpha^i`.
    //
    // `params.d` is dynamic so the compiler cannot unroll, but the
    // per-iter body is ~3 ALU ops + 1 16 B load; at D=32 the per-
    // thread cost is ~100 cycles, dominated by the load. The
    // alpha_pows buffer (512 B at D=32) fits comfortably in L1 /
    // texture cache and is read identically by all threads in a
    // simdgroup.
    for (uint i = 0; i < params.d; i++) {
        ulong bit = (cell >> i) & 1u;
        ulong mask = 0u - bit;  // 0 -> 0, 1 -> 0xFFFF...FF
        ulong2 pw = alpha_pows[i];
        acc.x ^= pw.x & mask;
        acc.y ^= pw.y & mask;
    }

    out[gid] = acc;
}
