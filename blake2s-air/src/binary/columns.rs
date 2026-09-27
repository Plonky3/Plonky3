use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use super::{G_PER_ROUND, NUM_ROUNDS};

/// Columns of one trace row, which proves one BLAKE2s compression.
///
/// Every word is stored as 32 bit columns, least significant bit first.
#[repr(C)]
pub struct Blake2sBinaryCols<T> {
    /// The chaining value `h[0..8]`, which initializes `v[0..8]`.
    pub chaining_value: [[T; 32]; 8],

    /// The sixteen message words `m[0..16]`.
    pub block: [[T; 32]; 16],

    /// Low 32 bits of the byte counter `t`, XORed into `v[12]`.
    pub counter_low: [T; 32],

    /// High 32 bits of the byte counter `t`, XORed into `v[13]`.
    pub counter_high: [T; 32],

    /// One bit, set on the final block of a message.
    ///
    /// When set, every bit of `v[14]` is inverted.
    ///
    /// One cell leaves only the all-zero and all-one words reachable, as RFC 7693 requires.
    pub last_block: T,

    /// Witness of every G step, indexed by round then by step within the round.
    pub rounds: [[Blake2sBinaryGCols<T>; G_PER_ROUND]; NUM_ROUNDS],
}

/// Witness columns of one G step.
///
/// G mixes the words `(a, b, c, d)` with two message words `m_x` and `m_y`:
///
/// ```text
///     a_1 = a + b + m_x          d_1 = (d ^ a_1) >>> 16
///     c_1 = c + d_1              b_1 = (b ^ c_1) >>> 12
///     a_2 = a_1 + b_1 + m_y      d_2 = (d_1 ^ a_2) >>> 8
///     c_2 = c_1 + d_2            b_2 = (b_1 ^ c_2) >>> 7
/// ```
///
/// Only `b_1`, `d_1`, `b_2`, `d_2` and the carries are stored.
///
/// The other words are XORs of stored words, so they cost no column:
///
/// ```text
///     a_1 = d   ^ (d_1 <<< 16)
///     c_1 = b   ^ (b_1 <<< 12)
///     a_2 = d_1 ^ (d_2 <<< 8)
///     c_2 = b_1 ^ (b_2 <<< 7)
/// ```
#[repr(C)]
pub struct Blake2sBinaryGCols<T> {
    /// Carries into bits `1..32` of `a + b`.
    pub add1_carries: [T; 31],

    /// The word `d_1`.
    pub d1: [T; 32],

    /// The word `b_1`.
    pub b1: [T; 32],

    /// Carries into bits `1..32` of `a_1 + b_1`.
    pub add2_carries: [T; 31],

    /// The word `d_2`, which replaces `d` in the state.
    pub d2: [T; 32],

    /// The word `b_2`, which replaces `b` in the state.
    pub b2: [T; 32],
}

/// Number of main trace columns of the binary BLAKE2s AIR.
pub const NUM_BLAKE2S_BINARY_COLS: usize = size_of::<Blake2sBinaryCols<u8>>();

impl<T> Borrow<Blake2sBinaryCols<T>> for [T] {
    fn borrow(&self) -> &Blake2sBinaryCols<T> {
        // The row must hold exactly one set of columns.
        debug_assert_eq!(self.len(), NUM_BLAKE2S_BINARY_COLS);

        // Safety: the struct is `repr(C)` and made only of `T`, so it has the layout of `[T; N]`.
        let (prefix, shorts, suffix) = unsafe { self.align_to::<Blake2sBinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<Blake2sBinaryCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut Blake2sBinaryCols<T> {
        // The row must hold exactly one set of columns.
        debug_assert_eq!(self.len(), NUM_BLAKE2S_BINARY_COLS);

        // Safety: the struct is `repr(C)` and made only of `T`, so it has the layout of `[T; N]`.
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<Blake2sBinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
