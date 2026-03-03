use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

/// Columns for a Poseidon AIR which computes one permutation per row.
///
/// The columns of the STARK are divided into the three different round sections of the Poseidon
/// permutation: beginning full rounds, partial rounds, and ending full rounds. For the full
/// rounds we store an [`SBox`] columnset for each state variable, and for the partial rounds we
/// store only for the first state variable.
///
/// Unlike Poseidon2, partial rounds store a full `post: [T; WIDTH]` because the dense MDS matrix
/// fully mixes all elements each round. Carrying expressions forward without committing
/// intermediates would cause expression blowup.
#[repr(C)]
pub struct PoseidonCols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub inputs: [T; WIDTH],

    /// Beginning Full Rounds
    pub beginning_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],

    /// Partial Rounds
    pub partial_rounds: [PartialRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; PARTIAL_ROUNDS],

    /// Ending Full Rounds
    pub ending_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],
}

/// Full round columns.
#[repr(C)]
pub struct FullRound<T, const WIDTH: usize, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize> {
    /// Possible intermediate results within each S-box.
    pub sbox: [SBox<T, SBOX_DEGREE, SBOX_REGISTERS>; WIDTH],
    /// The post-state, i.e. the entire layer after this full round.
    pub post: [T; WIDTH],
}

/// Partial round columns.
///
/// Unlike Poseidon2, this stores a full `post: [T; WIDTH]` instead of just `post_sbox: T`.
/// The dense MDS matrix fully mixes all elements each round, so without committing the
/// full post-state, expression degrees would blow up.
#[repr(C)]
pub struct PartialRound<T, const WIDTH: usize, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize>
{
    /// Possible intermediate results within the S-box (applied to state[0] only).
    pub sbox: SBox<T, SBOX_DEGREE, SBOX_REGISTERS>,
    /// The full post-state after the MDS layer.
    pub post: [T; WIDTH],
}

/// Possible intermediate results within an S-box.
///
/// Use this column-set for an S-box that can be computed with `REGISTERS`-many intermediate results
/// (not counting the final output). The S-box is checked to ensure that `REGISTERS` is the optimal
/// number of registers for the given `DEGREE` for the degrees given in the Poseidon paper:
/// `3`, `5`, `7`, and `11`. See `eval_sbox` for more information.
#[repr(C)]
pub struct SBox<T, const DEGREE: u64, const REGISTERS: usize>(pub [T; REGISTERS]);

pub const fn num_cols<
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>() -> usize {
    size_of::<PoseidonCols<u8, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>>(
    )
}

impl<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Borrow<PoseidonCols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>>
    for [T]
{
    fn borrow(
        &self,
    ) -> &PoseidonCols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    {
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<PoseidonCols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> BorrowMut<PoseidonCols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>>
    for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut PoseidonCols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    {
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<PoseidonCols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
