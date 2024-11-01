use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};

use p3_util::indices_arr;

use crate::U32_LIMBS;

/// Note: The ordering of each array is based on the input mapping. As the spec says,
///
/// > The mapping between the bits of s and those of a is `s[w(5y + x) + z] = a[x][y][z]`.
///
/// Thus, for example, `a_prime` is stored in `y, x, z` order. This departs from the more common
/// convention of `x, y, z` order, but it has the benefit that input lists map to AIR columns in a
/// nicer way.
#[derive(Debug)]
#[repr(C)]
pub struct Blake3Cols<T> {
    /// The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    pub first_round: T,
    pub last_round: T,

    /// A register which indicates if a row should be exported, i.e. included in a multiset equality
    /// argument. Should be 1 only for certain rows which are final steps, i.e. with
    /// `step_flags[23] = 1`.
    pub export: T,

    pub block_words: [[T; U32_LIMBS]; 16],

    pub row0_input: [[T; U32_LIMBS]; 4],
    pub row1_input: [[T; 32]; 4],
    pub row2_input: [[T; U32_LIMBS]; 4],
    pub row3_input: [[T; 32]; 4],

    pub row0_prime: [[T; U32_LIMBS]; 4],
    pub row1_prime: [[T; 32]; 4],
    pub row2_prime: [[T; U32_LIMBS]; 4],
    pub row3_prime: [[T; 32]; 4],

    pub aux_columns: [[T; 8]; 4],

    pub row0_middle: [[T; U32_LIMBS]; 4],
    pub row1_middle: [[T; 32]; 4],
    pub row2_middle: [[T; U32_LIMBS]; 4],
    pub row3_middle: [[T; 32]; 4],

    pub row0_middle_prime: [[T; U32_LIMBS]; 4],
    pub row1_middle_prime: [[T; 32]; 4],
    pub row2_middle_prime: [[T; U32_LIMBS]; 4],
    pub row3_middle_prime: [[T; 32]; 4],

    pub aux_diagonals: [[T; 8]; 4],

    pub row0_output: [[T; U32_LIMBS]; 4],
    pub row1_output: [[T; 32]; 4],
    pub row2_output: [[T; U32_LIMBS]; 4],
    pub row3_output: [[T; 32]; 4],
}

pub const NUM_BLAKE3_COLS: usize = size_of::<Blake3Cols<u8>>();
pub(crate) const BLAKE3_COL_MAP: Blake3Cols<usize> = make_col_map();

const fn make_col_map() -> Blake3Cols<usize> {
    let indices_arr = indices_arr::<NUM_BLAKE3_COLS>();
    unsafe { transmute::<[usize; NUM_BLAKE3_COLS], Blake3Cols<usize>>(indices_arr) }
}

impl<T> Borrow<Blake3Cols<T>> for [T] {
    fn borrow(&self) -> &Blake3Cols<T> {
        debug_assert_eq!(self.len(), NUM_BLAKE3_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<Blake3Cols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<Blake3Cols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut Blake3Cols<T> {
        debug_assert_eq!(self.len(), NUM_BLAKE3_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<Blake3Cols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
