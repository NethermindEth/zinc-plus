// =====================================================================
// raa_f2_encode.metal — GPU kernels for the RAA F_2 encoder used by
// `RaaF2Code` (zip-plus/src/code/raa_f2.rs).
//
// Adapted from `albertgarreta/bolt-rs:src/metal_gpu/shaders/expander_inner_product.metal`
// (the `raa_repeat`, `raa_permute`, `raa_prefix_xor_block_phase{1,2,3}`
// kernels), with two substantive changes for the zinc-plus shape:
//
//   1. **Data type is `ulong` (u64), not `uint` (u32).** zinc-plus's
//      commit cells are `BinaryPoly<64>` = `#[repr(transparent)] u64`
//      (PACKED_STORAGE_WIDTH = 64). F_2 XOR over packed u64 polys is
//      bitwise XOR, the same operation as bolt's u32 version.
//
//   2. **Row-major layout, not column-major.** zinc-plus's
//      `DenseRowMatrix<Cw>::data` stores cell (r, c) at
//      `data[r * ncols + c]`. Each parallel codeword is contiguous in
//      memory (matches the CPU `RaaF2Code::encode_into_with_scratch`
//      shape exactly). Within-row prefix-XOR walks a contiguous span,
//      which is faster on Apple GPU's L1/L2 than bolt's column-major
//      stride-`n_rows` walk.
//
// Pipeline (called sequentially within one command buffer by the
// dispatcher in `metal_gpu/mod.rs`):
//   1. `raa_f2_repeat`              — replicate input row REP times.
//   2. `raa_f2_permute` (π₁)        — gather columns through π₁.
//   3. `raa_f2_prefix_xor_phase1`   — per-block sequential XOR-scan.
//   4. `raa_f2_prefix_xor_phase2`   — scan the per-row block carries.
//   5. `raa_f2_prefix_xor_phase3`   — apply carries to non-first blocks.
//   6. `raa_f2_permute` (π₂)        — gather through π₂.
//   7. `raa_f2_prefix_xor_phase1`   — second prefix-XOR.
//   8. `raa_f2_prefix_xor_phase2`
//   9. `raa_f2_prefix_xor_phase3`
//
// Caller contract for `raa_f2_repeat`:
//   * `data` must already contain the input row in the first `msg_len`
//     columns of each parallel-codeword row. The kernel replicates
//     those into the remaining `(REP - 1) * msg_len` columns. CPU side
//     does a memcpy of the input rows into `data[r * ncols + 0..msg_len]`
//     before dispatch.
// =====================================================================

#include <metal_stdlib>
using namespace metal;

/// Shared params struct for every RAA kernel. Individual kernels read
/// only the fields they need; the unused fields are simply ignored.
/// Living in one struct keeps the dispatcher's buffer-binding count
/// down — one shared params buffer for all nine encoders in a single
/// command buffer.
struct RaaParams {
    uint n_rows;     // number of parallel codewords
    uint ncols;      // codeword length (= msg_len * q)
    uint msg_len;    // input row length (= ncols / q)
    uint q;          // repetition factor (REP)
    uint block_size; // prefix-XOR block size (phase1/3 only)
    uint n_blocks;   // ceil(ncols / block_size) (phase2/3 only)
};

// -----------------------------------------------------------------
// raa_f2_repeat
//
// Row-major **block concatenation** layout (matches the CPU
// `RaaF2Code::encode_internal_with_scratch` which does
// `for _ in 0..REP { buf.extend_from_slice(&row); }`). After this
// kernel runs, the row is `q` adjacent copies of the input msg:
//   col 0..msg_len:           original input
//   col msg_len..2*msg_len:   copy of input
//   col 2*msg_len..3*msg_len: copy of input
//   ...
// i.e. `data[row * ncols + c] = data[row * ncols + (c % msg_len)]`
// for `c ∈ [msg_len, ncols)`.
//
// Bolt-rs's RAA uses an interleaved layout (col-major and within-col
// adjacent replication), which doesn't match zinc-plus's CPU
// codepath. Both end up with the same per-input multiplicity (q
// copies each), but the permutation indices `perm_1` were built
// assuming the block-concatenation layout, so the kernel has to
// match.
//
// Dispatch: `n_rows * (ncols - msg_len)` threads.
// Tid layout: tid = row * (ncols - msg_len) + (dst_col - msg_len).
// -----------------------------------------------------------------
kernel void raa_f2_repeat(
    device ulong*           data   [[buffer(0)]],
    constant RaaParams&     p      [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    uint per_row_to_fill = p.ncols - p.msg_len;
    uint total = p.n_rows * per_row_to_fill;
    if (tid >= total) return;

    uint row = tid / per_row_to_fill;
    uint dst_offset = tid % per_row_to_fill;   // 0..(ncols - msg_len)
    uint dst_col = p.msg_len + dst_offset;
    uint src_col = dst_col % p.msg_len;

    data[row * p.ncols + dst_col] = data[row * p.ncols + src_col];
}

// -----------------------------------------------------------------
// raa_f2_permute (out-of-place)
//
// `dst[row * ncols + col] = src[row * ncols + perm[col]]` for every
// (row, col) pair.
//
// Dispatch: `n_rows * ncols` threads.
// Tid layout: tid = row * ncols + col.
// -----------------------------------------------------------------
kernel void raa_f2_permute(
    device const ulong*     src    [[buffer(0)]],
    device ulong*           dst    [[buffer(1)]],
    device const uint*      perm   [[buffer(2)]],
    constant RaaParams&     p      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.n_rows * p.ncols;
    if (tid >= total) return;
    uint row = tid / p.ncols;
    uint col = tid % p.ncols;
    uint src_col = perm[col];
    dst[row * p.ncols + col] = src[row * p.ncols + src_col];
}

// -----------------------------------------------------------------
// raa_f2_prefix_xor_phase1
//
// Per-row block-scan phase 1: each thread sequentially XOR-scans one
// (row, block) strip of `block_size` columns, in place. Also writes
// the last element of each strip to
// `block_carries[row * n_blocks + block]` (row-major carry layout —
// each row's carries are contiguous).
//
// Dispatch: total threads = n_rows * n_blocks
// Tid layout: tid = row * n_blocks + block
// -----------------------------------------------------------------
kernel void raa_f2_prefix_xor_phase1(
    device ulong*           data           [[buffer(0)]],
    device ulong*           block_carries  [[buffer(1)]],
    constant RaaParams&     p              [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.n_rows * p.n_blocks;
    if (tid >= total) return;
    uint row = tid / p.n_blocks;
    uint block = tid % p.n_blocks;
    uint col_start = block * p.block_size;
    uint col_end = (block + 1) * p.block_size;
    if (col_end > p.ncols) col_end = p.ncols;

    // Sequential XOR-scan within the block — walks contiguous memory
    // in the row-major layout, hot in L1 after the first cache line.
    uint base = row * p.ncols;
    for (uint col = col_start + 1; col < col_end; col++) {
        data[base + col] ^= data[base + col - 1];
    }
    // Save the strip's running XOR (last element of the block) as the
    // block's carry, in row-major carry order.
    block_carries[row * p.n_blocks + block] = data[base + col_end - 1];
}

// -----------------------------------------------------------------
// raa_f2_prefix_xor_phase2
//
// Scan the per-row block carries in place. After this,
// `block_carries[row * n_blocks + block]` holds the XOR of all blocks
// 0..=block for that row.
//
// Dispatch: n_rows threads (one per row).
// -----------------------------------------------------------------
kernel void raa_f2_prefix_xor_phase2(
    device ulong*           block_carries [[buffer(0)]],
    constant RaaParams&     p             [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n_rows) return;
    uint base = tid * p.n_blocks;
    for (uint block = 1; block < p.n_blocks; block++) {
        block_carries[base + block] ^= block_carries[base + block - 1];
    }
}

// -----------------------------------------------------------------
// raa_f2_prefix_xor_phase3
//
// Apply the (block - 1) carry to every element of `block`. Block 0 has
// no carry to apply and is skipped (its prefix-XOR was completed by
// phase 1).
//
// Dispatch: total threads = n_rows * ncols (one per element).
// Tid layout: tid = row * ncols + col.
// -----------------------------------------------------------------
kernel void raa_f2_prefix_xor_phase3(
    device ulong*           data           [[buffer(0)]],
    device const ulong*     block_carries  [[buffer(1)]],
    constant RaaParams&     p              [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.n_rows * p.ncols;
    if (tid >= total) return;
    uint row = tid / p.ncols;
    uint col = tid % p.ncols;
    uint block = col / p.block_size;
    if (block == 0) return;  // first block already complete
    data[row * p.ncols + col] ^= block_carries[row * p.n_blocks + (block - 1)];
}
