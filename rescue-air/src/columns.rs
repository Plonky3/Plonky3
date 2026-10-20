use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

/// Columns for a Rescue AIR computing one permutation per row.
///
/// Both MDS layers are linear, so they need no witness of their own; what has
/// to be committed is each S-box output, because the inverse S-box is not a
/// low-degree polynomial in its input.
#[repr(C)]
pub struct RescueCols<T, const WIDTH: usize, const NUM_ROUNDS: usize> {
    /// The permutation input, i.e. the state before the first round.
    pub inputs: [T; WIDTH],

    pub rounds: [RescueRound<T, WIDTH>; NUM_ROUNDS],
}

/// One Rescue round: forward S-box, MDS and constants, then inverse S-box,
/// MDS and constants.
#[repr(C)]
pub struct RescueRound<T, const WIDTH: usize> {
    /// `state^ALPHA`, the forward S-box output.
    pub sbox_pow: [T; WIDTH],
    /// The state after the first MDS layer and its constants.
    pub mid: [T; WIDTH],
    /// `mid^(1/ALPHA)`, the inverse S-box output.
    pub inv_sbox: [T; WIDTH],
    /// The state after the second MDS layer and its constants, i.e. the input
    /// to the next round.
    pub post: [T; WIDTH],
}

pub const fn num_cols<const WIDTH: usize, const NUM_ROUNDS: usize>() -> usize {
    size_of::<RescueCols<u8, WIDTH, NUM_ROUNDS>>()
}

impl<T, const WIDTH: usize, const NUM_ROUNDS: usize> Borrow<RescueCols<T, WIDTH, NUM_ROUNDS>>
    for [T]
{
    fn borrow(&self) -> &RescueCols<T, WIDTH, NUM_ROUNDS> {
        debug_assert_eq!(self.len(), num_cols::<WIDTH, NUM_ROUNDS>());
        let (prefix, shorts, suffix) =
            unsafe { self.align_to::<RescueCols<T, WIDTH, NUM_ROUNDS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize, const NUM_ROUNDS: usize> BorrowMut<RescueCols<T, WIDTH, NUM_ROUNDS>>
    for [T]
{
    fn borrow_mut(&mut self) -> &mut RescueCols<T, WIDTH, NUM_ROUNDS> {
        debug_assert_eq!(self.len(), num_cols::<WIDTH, NUM_ROUNDS>());
        let (prefix, shorts, suffix) =
            unsafe { self.align_to_mut::<RescueCols<T, WIDTH, NUM_ROUNDS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
