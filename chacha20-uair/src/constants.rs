//! ChaCha20 constants.

/// The ChaCha20 constant words: "expand 32-byte k" in little-endian u32s.
pub const CHACHA_CONSTANTS: [u32; 4] = [
    0x6170_7865, // "expa"
    0x3320_646e, // "nd 3"
    0x7962_2d32, // "2-by"
    0x6b20_6574, // "te k"
];

/// Rotation amounts for each step of a quarter round.
/// Step 0: <<< 16, Step 1: <<< 12, Step 2: <<< 8, Step 3: <<< 7.
pub const QR_ROTATIONS: [u32; 4] = [16, 12, 8, 7];
