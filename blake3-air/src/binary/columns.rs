use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use super::{G_PER_ROUND, NUM_ROUNDS};

/// Columns for a binary Blake-3 AIR which computes one compression per row.
///
/// Words are stored as 32 bits, least significant bit first.
#[repr(C)]
pub struct Blake3BinaryCols<T> {
    /// The chaining value, which initializes `v[0..8]`.
    pub chaining_value: [[T; 32]; 8],

    /// The message words.
    pub block: [[T; 32]; 16],

    /// Low half of the block counter, `v[12]`.
    pub counter_low: [T; 32],

    /// High half of the block counter, `v[13]`.
    pub counter_high: [T; 32],

    /// Number of message bytes in the block, `v[14]`.
    pub block_len: [T; 32],

    /// Domain separation flags, `v[15]`.
    pub flags: [T; 32],

    /// Witness columns for every G step, indexed by round then step.
    pub rounds: [[Blake3BinaryGCols<T>; G_PER_ROUND]; NUM_ROUNDS],
}

/// Witness columns for one G step.
///
/// G mixes `(a, b, c, d)` with the message words `mx` and `my`:
///
/// ```text
/// a1 = a + b + mx;   d1 = (d ^ a1) >>> 16;   c1 = c + d1;   b1 = (b ^ c1) >>> 12;
/// a2 = a1 + b1 + my; d2 = (d1 ^ a2) >>> 8;   c2 = c1 + d2;  b2 = (b1 ^ c2) >>> 7;
/// ```
///
/// Only `b1`, `d1`, `b2`, `d2` are stored. The other words are linear in them:
/// `a1 = d ^ (d1 <<< 16)`, `c1 = b ^ (b1 <<< 12)`, `a2 = d1 ^ (d2 <<< 8)` and
/// `c2 = b1 ^ (b2 <<< 7)`.
#[repr(C)]
pub struct Blake3BinaryGCols<T> {
    /// Carries into bits `1..32` of `a + b`.
    pub add1_carries: [T; 31],

    /// The word `d1`.
    pub d1: [T; 32],

    /// The word `b1`.
    pub b1: [T; 32],

    /// Carries into bits `1..32` of `a1 + b1`.
    pub add2_carries: [T; 31],

    /// The output word `d2`.
    pub d2: [T; 32],

    /// The output word `b2`.
    pub b2: [T; 32],
}

/// Number of main trace columns of the binary Blake-3 AIR.
pub const NUM_BLAKE3_BINARY_COLS: usize = size_of::<Blake3BinaryCols<u8>>();

impl<T> Borrow<Blake3BinaryCols<T>> for [T] {
    fn borrow(&self) -> &Blake3BinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_BLAKE3_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<Blake3BinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<Blake3BinaryCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut Blake3BinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_BLAKE3_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<Blake3BinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
