use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use super::{G_PER_ROUND, NUM_ROUNDS};

/// Columns for a binary BLAKE2s AIR which computes one compression per row.
///
/// Words are stored as 32 bits, least significant bit first.
#[repr(C)]
pub struct Blake2sBinaryCols<T> {
    /// The chaining value, which initializes `v[0..8]`.
    pub chaining_value: [[T; 32]; 8],

    /// The message words.
    pub block: [[T; 32]; 16],

    /// Low half of the byte counter, XORed into `v[12]`.
    pub counter_low: [T; 32],

    /// High half of the byte counter, XORed into `v[13]`.
    pub counter_high: [T; 32],

    /// The last-block flag, XORed into `v[14]`. All ones on the final block.
    pub last_block: [T; 32],

    /// The last-node flag, XORed into `v[15]`. All ones for the last node of a tree.
    pub last_node: [T; 32],

    /// Witness columns for every G step, indexed by round then step.
    pub rounds: [[Blake2sBinaryGCols<T>; G_PER_ROUND]; NUM_ROUNDS],
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
pub struct Blake2sBinaryGCols<T> {
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

/// Number of main trace columns of the binary BLAKE2s AIR.
pub const NUM_BLAKE2S_BINARY_COLS: usize = size_of::<Blake2sBinaryCols<u8>>();

impl<T> Borrow<Blake2sBinaryCols<T>> for [T] {
    fn borrow(&self) -> &Blake2sBinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_BLAKE2S_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<Blake2sBinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<Blake2sBinaryCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut Blake2sBinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_BLAKE2S_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<Blake2sBinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
