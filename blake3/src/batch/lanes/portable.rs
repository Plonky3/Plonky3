//! One lane per plain `u32`, for targets without a vector backend.

use blake3::{BLOCK_LEN, OUT_LEN};

use super::Word;
use crate::batch::compress::{BLOCK_WORDS, STATE_WORDS};

/// One state or message word for a single lane.
pub(super) type Vector = u32;

/// Lanes in one register.
pub(super) const WIDTH: usize = 1;

/// Independent register groups hashed together.
///
/// One state already fills a general-purpose register file, so a second only spills.
pub(super) const GROUPS: usize = 1;

impl Word for u32 {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        value
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        self & rhs
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        self.rotate_right(16)
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        self.rotate_right(12)
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        self.rotate_right(8)
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        self.rotate_right(7)
    }
}

/// Read one block as sixteen little-endian words.
#[inline(always)]
pub(super) fn load_block([row]: &[&[u8; BLOCK_LEN]; WIDTH]) -> [u32; BLOCK_WORDS] {
    let (words, _) = row.as_chunks::<4>();
    core::array::from_fn(|w| u32::from_le_bytes(words[w]))
}

/// Write one chaining value as a little-endian digest.
#[inline(always)]
pub(super) fn store_digests(state: &[u32; STATE_WORDS], [out]: &mut [[u8; OUT_LEN]; WIDTH]) {
    for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *bytes = word.to_le_bytes();
    }
}
