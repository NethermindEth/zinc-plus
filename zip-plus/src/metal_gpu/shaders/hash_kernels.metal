// =====================================================================
// hash_kernels.metal — Blake3 leaf-hash kernel for the commit phase.
//
// Adapted from `albertgarreta/bolt-rs:src/metal_gpu/shaders/hash_kernels.metal`.
// Stripped to only the Blake3 path (the other hash families are not used
// by zinc-plus's commit phase, which fixes the Merkle hash to Blake3 via
// `MerkleTree` in `zip-plus/src/merkle.rs`).
//
// Layout:
//   * `data`         — flat bytes: `num_cols * col_byte_len` bytes.
//     Column `i` starts at `i * col_byte_len` and has `col_byte_len` bytes.
//   * `out`          — flat bytes: `num_cols * 32` bytes.
//     Digest for column `i` is written at `i * 32`.
//   * `params.num_cols`      — number of columns to hash.
//   * `params.col_byte_len`  — bytes per column (hash input length).
//
// Thread grid:
//   * Launch `num_cols` threads. Thread `gid` hashes column `gid`.
//
// Correctness: each leaf hash matches `blake3::hash(&data[i*len .. (i+1)*len])`
// byte-for-byte. This is what `zip-plus::merkle::hash_grouped_leaf_from_row_major`
// produces on the CPU side. The CPU-side tree builder then consumes these
// leaves unchanged.
// =====================================================================

#include <metal_stdlib>
using namespace metal;

struct HashParams {
    uint num_cols;
    uint col_byte_len;
};

static inline uint rotr32(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

// =====================================================================
// Blake3 (single- and multi-chunk path, up to 64 chunks = 64 KB inputs)
// =====================================================================

constant uint BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

constant uchar BLAKE3_SIGMA[7][16] = {
    {  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {  2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8 },
    {  3, 4,10,12,13, 2, 7,14, 6, 5, 9, 0,11,15, 8, 1 },
    { 10, 7,12, 9,14, 3,13,15, 4, 0,11, 2, 5, 8, 1, 6 },
    { 12,13, 9,11,15,10,14, 8, 7, 2, 5, 3, 0, 1, 6, 4 },
    {  9,14,11, 5, 8,12,15, 1,13, 3, 0,10, 2, 6, 4, 7 },
    { 11,15, 5, 0, 1, 9, 8, 6,14,10, 2,12, 3, 4, 7,13 }
};

#define BLAKE3_CHUNK_START       (1u << 0)
#define BLAKE3_CHUNK_END         (1u << 1)
#define BLAKE3_ROOT              (1u << 3)

static inline void blake3_g(thread uint* v, uint a, uint b, uint c, uint d, uint mx, uint my) {
    v[a] = v[a] + v[b] + mx;
    v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + my;
    v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7);
}

// Compress one 64-byte block. `cv` is the input chaining value, output
// overwrites `cv` (partial update) or writes a 16-word output if root.
static inline void blake3_compress(
    thread uint* cv,
    thread const uint* block_words,
    ulong counter,
    uint block_len,
    uint flags,
    thread uint* out16 /* optional, length 16, can be null-equivalent */)
{
    uint v[16];
    for (uint i = 0; i < 8; ++i) v[i] = cv[i];
    for (uint i = 0; i < 4; ++i) v[8 + i] = BLAKE3_IV[i];
    v[12] = (uint)(counter & 0xffffffffu);
    v[13] = (uint)(counter >> 32);
    v[14] = block_len;
    v[15] = flags;

    uint m[16];
    for (uint i = 0; i < 16; ++i) m[i] = block_words[i];

    for (uint r = 0; r < 7; ++r) {
        constant uchar* s = BLAKE3_SIGMA[r];
        uint m2[16];
        for (uint i = 0; i < 16; ++i) m2[i] = m[s[i]];
        blake3_g(v, 0, 4,  8, 12, m2[ 0], m2[ 1]);
        blake3_g(v, 1, 5,  9, 13, m2[ 2], m2[ 3]);
        blake3_g(v, 2, 6, 10, 14, m2[ 4], m2[ 5]);
        blake3_g(v, 3, 7, 11, 15, m2[ 6], m2[ 7]);
        blake3_g(v, 0, 5, 10, 15, m2[ 8], m2[ 9]);
        blake3_g(v, 1, 6, 11, 12, m2[10], m2[11]);
        blake3_g(v, 2, 7,  8, 13, m2[12], m2[13]);
        blake3_g(v, 3, 4,  9, 14, m2[14], m2[15]);
    }

    if (out16 != nullptr) {
        for (uint i = 0; i < 8; ++i) {
            out16[i]     = v[i] ^ v[i+8];
            out16[i+8]   = v[i+8] ^ cv[i];
        }
    } else {
        for (uint i = 0; i < 8; ++i) cv[i] = v[i] ^ v[i+8];
    }
}

// Compress one 1024-byte chunk into an 8-word chaining value.
// `msg` points to the chunk bytes, `chunk_len` ≤ 1024 (only last chunk
// can be short). `chunk_counter` is the 0-based chunk index.
//
// `base_flags` is applied to every block. `is_root_chunk` is true only
// for single-chunk inputs where this IS the root compression — in that
// case ROOT is added to the flags of the LAST block only (Blake3 spec:
// ROOT is only set on the final compression that produces the output).
static inline void blake3_compress_chunk(
    device const uchar* msg,
    uint chunk_len,
    ulong chunk_counter,
    uint base_flags,
    bool is_root_chunk,
    thread uint* out_cv)
{
    uint cv[8];
    for (uint i = 0; i < 8; ++i) cv[i] = BLAKE3_IV[i];

    uint blk_words[16];
    uint off = 0;
    uint n_blocks_full = chunk_len / 64;
    uint rem = chunk_len - n_blocks_full * 64;
    uint total_blocks = n_blocks_full + ((rem > 0 || n_blocks_full == 0) ? 1 : 0);

    for (uint b = 0; b < total_blocks; ++b) {
        bool is_last = (b == total_blocks - 1);
        uint blk_len;
        if (!is_last || rem == 0) {
            if (b < n_blocks_full) {
                blk_len = 64;
                for (uint j = 0; j < 16; ++j) {
                    blk_words[j] = ((uint)msg[off + j*4 + 0]      )
                                 | ((uint)msg[off + j*4 + 1] <<  8)
                                 | ((uint)msg[off + j*4 + 2] << 16)
                                 | ((uint)msg[off + j*4 + 3] << 24);
                }
                off += 64;
            } else {
                blk_len = 0;
                for (uint j = 0; j < 16; ++j) blk_words[j] = 0;
            }
        } else {
            blk_len = rem;
            uchar tmp[64];
            for (uint i = 0; i < rem; ++i) tmp[i] = msg[off + i];
            for (uint i = rem; i < 64; ++i) tmp[i] = 0;
            for (uint j = 0; j < 16; ++j) {
                blk_words[j] = ((uint)tmp[j*4 + 0]      )
                             | ((uint)tmp[j*4 + 1] <<  8)
                             | ((uint)tmp[j*4 + 2] << 16)
                             | ((uint)tmp[j*4 + 3] << 24);
            }
            off += rem;
        }

        uint flags = base_flags;
        if (b == 0)                     flags |= BLAKE3_CHUNK_START;
        if (is_last)                    flags |= BLAKE3_CHUNK_END;
        if (is_last && is_root_chunk)   flags |= BLAKE3_ROOT;

        blake3_compress(cv, blk_words, chunk_counter, blk_len, flags, nullptr);
    }

    for (uint i = 0; i < 8; ++i) out_cv[i] = cv[i];
}

// Parent-compress two child CVs into one. `flags` should include PARENT
// (1u<<2); the caller adds ROOT (1u<<3) on the topmost compression only.
// Output is written to `out_cv` (length 8).
static inline void blake3_parent_compress(
    thread const uint* left_cv,
    thread const uint* right_cv,
    uint flags,
    thread uint* out_cv)
{
    uint blk[16];
    for (uint j = 0; j < 8; ++j) blk[j]     = left_cv[j];
    for (uint j = 0; j < 8; ++j) blk[j + 8] = right_cv[j];
    uint cv[8];
    for (uint j = 0; j < 8; ++j) cv[j] = BLAKE3_IV[j];
    blake3_compress(cv, blk, 0, 64, flags, nullptr);
    for (uint j = 0; j < 8; ++j) out_cv[j] = cv[j];
}

kernel void hash_leaves_blake3(
    device const uchar*        data   [[ buffer(0) ]],
    device uchar*              out    [[ buffer(1) ]],
    constant HashParams&       params [[ buffer(2) ]],
    uint                       gid    [[ thread_position_in_grid ]])
{
    if (gid >= params.num_cols) return;
    uint len = params.col_byte_len;
    device const uchar* msg = data + (uint)gid * len;

    // Single-chunk short-path: leaves ≤ 1 KB hit here.
    // is_root_chunk=true → ROOT is applied only to the LAST block of
    // the chunk, not every block.
    if (len <= 1024) {
        uint cv[8];
        blake3_compress_chunk(msg, len, 0, 0u, /*is_root_chunk=*/true, cv);
        device uchar* dst = out + (uint)gid * 32u;
        for (uint i = 0; i < 8; ++i) {
            dst[i*4 + 0] = (uchar)(cv[i]      );
            dst[i*4 + 1] = (uchar)(cv[i] >>  8);
            dst[i*4 + 2] = (uchar)(cv[i] >> 16);
            dst[i*4 + 3] = (uchar)(cv[i] >> 24);
        }
        return;
    }

    // ==== Multi-chunk path: streaming subtree-stack algorithm ====
    //
    // Replaces the previous bounded (≤64-chunk) path with Blake3's
    // standard streaming pattern. After each chunk, merge upward into
    // the parent subtree stack as long as the top has the same height
    // (mimicking how blake3's Hasher consumes input). The stack holds
    // at most `ceil(log2(n_chunks))` partial CVs, each 8 u32 = 32 B,
    // so even for 16 GB leaves (~16M chunks → depth 24) the per-thread
    // stack is < 1 KB.
    //
    // ROOT-flag rule (Blake3 spec): only the LAST compression that
    // produces the final 32-byte output carries ROOT. Two cases:
    //   1. Streaming merge: the last chunk's bubble-up may end with
    //      a merge that empties the stack — that's the root.
    //   2. Finalize merge: after the streaming loop, multiple subtree
    //      CVs may remain on the stack at different heights. We merge
    //      them top-down; the final merge gets ROOT.
    uint n_chunks = (len + 1023) / 1024;

    // 24-deep stack: supports leaves up to 2^24 × 1024 B = 16 GB.
    // SHA-256 F_2 at nvars=20 needs depth 14 (14 MB leaves); nvars=24
    // would need depth 18. 24 leaves comfortable headroom.
    const uint MAX_STACK_DEPTH = 24;
    uint stack[MAX_STACK_DEPTH * 8];
    uint stack_h[MAX_STACK_DEPTH];
    uint depth = 0;

    uint off = 0;
    for (uint c = 0; c < n_chunks; ++c) {
        bool is_last_chunk = (c + 1 == n_chunks);
        uint this_len = is_last_chunk ? (len - off) : 1024;
        uint cv[8];
        blake3_compress_chunk(msg + off, this_len, (ulong)c, 0u,
                              /*is_root_chunk=*/false, cv);
        off += this_len;

        // Bubble up: merge with same-height subtrees on the stack.
        uint cv_h = 0;
        while (depth > 0 && stack_h[depth - 1] == cv_h) {
            uint left[8];
            for (uint j = 0; j < 8; ++j) left[j] = stack[(depth - 1) * 8 + j];
            depth -= 1;
            // If this is the LAST chunk and the bubble-up emptied the
            // stack, this merge produces the root — set ROOT now.
            uint flags = (1u << 2);
            if (is_last_chunk && depth == 0) flags |= BLAKE3_ROOT;
            uint merged[8];
            blake3_parent_compress(left, cv, flags, merged);
            for (uint j = 0; j < 8; ++j) cv[j] = merged[j];
            cv_h += 1;
        }
        for (uint j = 0; j < 8; ++j) stack[depth * 8 + j] = cv[j];
        stack_h[depth] = cv_h;
        depth += 1;
    }

    // ==== Finalize: merge any remaining subtrees top-down ====
    //
    // After the streaming loop the stack holds 1+ CVs at distinct,
    // strictly-decreasing heights (largest subtree on bottom). Merge
    // the topmost two repeatedly until depth=1. The final merge in
    // this loop produces the root.
    while (depth > 1) {
        uint right[8];
        for (uint j = 0; j < 8; ++j) right[j] = stack[(depth - 1) * 8 + j];
        depth -= 1;
        uint left[8];
        for (uint j = 0; j < 8; ++j) left[j] = stack[(depth - 1) * 8 + j];
        uint flags = (1u << 2);
        if (depth == 1) flags |= BLAKE3_ROOT;
        uint merged[8];
        blake3_parent_compress(left, right, flags, merged);
        for (uint j = 0; j < 8; ++j) stack[(depth - 1) * 8 + j] = merged[j];
    }

    device uchar* dst = out + (uint)gid * 32u;
    for (uint i = 0; i < 8; ++i) {
        dst[i*4 + 0] = (uchar)(stack[i]      );
        dst[i*4 + 1] = (uchar)(stack[i] >>  8);
        dst[i*4 + 2] = (uchar)(stack[i] >> 16);
        dst[i*4 + 3] = (uchar)(stack[i] >> 24);
    }
}
