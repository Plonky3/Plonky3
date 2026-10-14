use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use p3_field::Field;

use super::super::G_PER_ROUND;
use super::super::columns::Blake3BinaryGCols;
use super::super::generation::round_compression_output;

/// Columns for one round of the compact binary Blake-3 AIR.
///
/// Every word is stored as 32 bits, least significant bit first. Compression groups
/// occupy eight consecutive rows; the seventh row contains the final round.
#[repr(C)]
pub struct Blake3BinaryCompactCols<T> {
    /// Original chaining value, retained for the output feed-forward.
    pub chaining_value: [[T; 32]; 8],
    /// Message words in the permutation used by this round.
    pub block: [[T; 32]; 16],
    /// State entering this round. On the first row, words 12..16 are the counter
    /// halves, block length and flags.
    pub state: [[T; 32]; 16],
    /// Witness columns for the eight G steps of this round.
    pub round: [Blake3BinaryGCols<T>; G_PER_ROUND],
}

/// Main trace width of the compact binary Blake-3 AIR.
pub const NUM_BLAKE3_BINARY_COMPACT_COLS: usize = size_of::<Blake3BinaryCompactCols<u8>>();

impl<T> Borrow<Blake3BinaryCompactCols<T>> for [T] {
    fn borrow(&self) -> &Blake3BinaryCompactCols<T> {
        assert_eq!(self.len(), NUM_BLAKE3_BINARY_COMPACT_COLS);
        // SAFETY: repr(C) contains only arrays of T, so its alignment is T's and
        // its size is exactly the checked number of cells.
        let (prefix, rows, suffix) = unsafe { self.align_to::<Blake3BinaryCompactCols<T>>() };
        assert!(prefix.is_empty() && suffix.is_empty());
        assert_eq!(rows.len(), 1);
        &rows[0]
    }
}

impl<T> BorrowMut<Blake3BinaryCompactCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut Blake3BinaryCompactCols<T> {
        assert_eq!(self.len(), NUM_BLAKE3_BINARY_COMPACT_COLS);
        // SAFETY: as above; the mutable slice gives exclusive access to the row.
        let (prefix, rows, suffix) = unsafe { self.align_to_mut::<Blake3BinaryCompactCols<T>>() };
        assert!(prefix.is_empty() && suffix.is_empty());
        assert_eq!(rows.len(), 1);
        &mut rows[0]
    }
}

impl<F: Field> Blake3BinaryCompactCols<F> {
    /// Recover all sixteen compression output words from the final round.
    ///
    /// Call this on logical row `8 * compression_index + 6`, identified by
    /// [`super::BLAKE3_BINARY_COMPACT_OUTPUT_ROW`]. Other rows return that round's
    /// output feed-forward, which is not the full compression result.
    ///
    /// # Panics
    ///
    /// Panics if a cell read is not a bit.
    pub fn compression_output(&self) -> [u32; 16] {
        round_compression_output(&self.round, &self.chaining_value)
    }
}
