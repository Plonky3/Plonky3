//! Numerical constants used by the SHA-256 compression function.

/// Number of 32-bit words in a SHA-256 message block.
///
/// One block carries `16 * 32 = 512` input bits.
pub const BLOCK_WORDS: usize = 16;

/// Number of compression rounds per block.
///
/// Fixed by the standard at 64.
pub const NUM_COMPRESSION_ROUNDS: usize = 64;

/// Number of new words produced by the message schedule.
///
/// Derived as `64 - 16 = 48`:
/// - 16 words come from the block input.
/// - 48 words are expanded from the recurrence.
pub const SCHEDULE_EXTENSIONS: usize = NUM_COMPRESSION_ROUNDS - BLOCK_WORDS;

/// Number of 32-bit words in the compression working state.
///
/// The working variables are `a, b, c, d, e, f, g, h`.
pub const STATE_WORDS: usize = 8;

/// Number of bits in a single SHA-256 word.
pub const WORD_BITS: usize = 32;

/// Bit width of one packed limb.
///
/// Chosen as 16 so a 32-bit word fits in two limbs and multiplications of two
/// limbs cannot overflow a 32-bit accumulator.
pub const BITS_PER_LIMB: usize = 16;

/// Number of 16-bit limbs needed to hold a 32-bit word.
pub const U32_LIMBS: usize = WORD_BITS / BITS_PER_LIMB;

/// Round constants used by the compression loop.
///
/// # Source
///
/// - Specified in FIPS 180-4, Section 4.2.2.
/// - Each value is the top 32 bits of the fractional part of the cube root of
///   one of the first 64 primes starting at 2.
///
/// # Usage
///
/// - The trace generator reads these values to compute `T_1`.
/// - The AIR injects them as constant expressions per round index.
pub const SHA256_K: [u32; NUM_COMPRESSION_ROUNDS] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Initial hash value used when hashing a fresh message.
///
/// # Source
///
/// - Specified in FIPS 180-4, Section 5.3.3.
/// - Each value is the top 32 bits of the fractional part of the square root
///   of one of the first 8 primes starting at 2.
///
/// # Usage
///
/// - Not consumed by the AIR itself.
/// - Exposed so tests and callers can pin a canonical starting state.
pub const SHA256_IV: [u32; STATE_WORDS] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Split a 32-bit word into little-endian 16-bit limbs.
///
/// # Arguments
///
/// - `value`: the 32-bit integer to split.
///
/// # Returns
///
/// A two-element array `[lo, hi]`:
/// - `lo` holds bits 0..16.
/// - `hi` holds bits 16..32.
///
/// # Examples
///
/// ```text
///     0xAABBCCDD --> [0xCCDD, 0xAABB]
/// ```
#[inline]
pub(crate) const fn u32_to_limbs(value: u32) -> [u16; U32_LIMBS] {
    // Truncate to the low 16 bits.
    let lo = value as u16;
    // Shift out the low 16 bits before truncation to get bits 16..32.
    let hi = (value >> BITS_PER_LIMB) as u16;
    [lo, hi]
}
