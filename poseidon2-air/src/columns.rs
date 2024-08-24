use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

/// Columns for Single-Row Poseidon2 STARK
///
/// The columns of the STARK are divided into the three different round sections of the Poseidon2
/// Permutation: beginning full rounds, partial rounds, and ending full rounds. For the full
/// rounds we store an [`SBox`] columnset for each state variable, and for the partial rounds we
/// store only for the first state variable. Because the matrix multiplications are linear
/// functions, we need only keep auxiliary columns for the S-BOX computations.
#[repr(C)]
pub struct Poseidon2Cols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub export: T,

    pub inputs: [T; WIDTH],

    /// Beginning Full Rounds
    pub beginning_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],

    /// Partial Rounds
    pub partial_rounds: [PartialRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; PARTIAL_ROUNDS],

    /// Ending Full Rounds
    pub ending_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],
}

/// Full Round Columns
#[repr(C)]
pub struct FullRound<T, const WIDTH: usize, const SBOX_DEGREE: usize, const SBOX_REGISTERS: usize> {
    /// S-BOX Columns
    pub sbox: [SBox<T, SBOX_DEGREE, SBOX_REGISTERS>; WIDTH],
}

/// Partial Round Columns
#[repr(C)]
pub struct PartialRound<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
> {
    /// S-BOX Columns
    pub sbox: SBox<T, SBOX_DEGREE, SBOX_REGISTERS>,
}

/// S-BOX Columns
///
/// Use this column-set for an S-BOX that can be computed in `REGISTERS`-many columns. The S-BOX is
/// checked to ensure that `REGISTERS` is the optimal number of registers for the given `DEGREE`
/// for the degrees given in the Poseidon2 paper: `3`, `5`, `7`, and `11`. See [`Self::eval`] for
/// more information.
#[repr(C)]
pub struct SBox<T, const DEGREE: usize, const REGISTERS: usize>(pub [T; REGISTERS]);

pub const fn num_cols<
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>() -> usize {
    size_of::<Poseidon2Cols<u8, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>>(
    )
}

pub const fn make_col_map<
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>() -> Poseidon2Cols<usize, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> {
    todo!()
    // let indices_arr = indices_arr::<
    //     { num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>() },
    // >();
    // unsafe {
    //     transmute::<
    //         [usize;
    //             num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()],
    //         Poseidon2Cols<
    //             usize,
    //             WIDTH,
    //             SBOX_DEGREE,
    //             SBOX_REGISTERS,
    //             HALF_FULL_ROUNDS,
    //             PARTIAL_ROUNDS,
    //         >,
    //     >(indices_arr)
    // }
}

impl<
        T,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Borrow<Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>>
    for [T]
{
    fn borrow(
        &self,
    ) -> &Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<Poseidon2Cols<
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
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >
    BorrowMut<
        Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    > for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<Poseidon2Cols<
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
