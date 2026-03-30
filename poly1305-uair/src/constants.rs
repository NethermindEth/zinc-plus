//! Poly1305 constants.

/// The Poly1305 prime: p = 2^130 - 5.
///
/// In 32-bit little-endian limbs: [0xFFFF_FFFB, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 3].
pub const POLY1305_P: [u32; 5] = [0xFFFF_FFFB, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 3];

/// Clamping mask for r (per RFC 7539).
///
/// Bytes r[3],r[7],r[11],r[15] have top 4 bits cleared;
/// bytes r[4],r[8],r[12] have bottom 2 bits cleared.
///
/// In 32-bit LE words: r[0] &= 0x0FFF_FFFF, r[1..3] &= 0x0FFF_FFFC.
pub const R_CLAMP_MASKS: [u32; 4] = [0x0FFF_FFFF, 0x0FFF_FFFC, 0x0FFF_FFFC, 0x0FFF_FFFC];
