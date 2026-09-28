use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use crate::NUM_ROUNDS;

/// Number of trace rows per permutation: one input row per round, then the output row.
pub const KECCAK_BINARY_ROWS_PER_PERM: usize = NUM_ROUNDS + 1;

/// One row of the characteristic-2 Keccak-f AIR.
///
/// Row `r < 24` of a permutation holds the input state of round `r`.
/// Row `24` holds the permutation output.
/// Padding rows after the last permutation are output rows with an all-zero state.
#[derive(Debug)]
#[repr(C)]
pub struct KeccakBinaryCols<T> {
    /// One-hot row kind.
    ///
    /// `round_flags[r] = 1` on the input row of round `r < 24`.
    /// `round_flags[24] = 1` on an output or padding row.
    pub round_flags: [T; KECCAK_BINARY_ROWS_PER_PERM],

    /// State bits in y-major order: `a[y][x][z]` is bit `z` (LSB first) of lane `5y + x`.
    pub a: [[[T; 64]; 5]; 5],
}

/// Number of columns of the characteristic-2 Keccak-f AIR.
pub const NUM_KECCAK_BINARY_COLS: usize = size_of::<KeccakBinaryCols<u8>>();

impl<T> Borrow<KeccakBinaryCols<T>> for [T] {
    fn borrow(&self) -> &KeccakBinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_KECCAK_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<KeccakBinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<KeccakBinaryCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut KeccakBinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_KECCAK_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<KeccakBinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
