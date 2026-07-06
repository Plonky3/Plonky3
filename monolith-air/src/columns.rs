//! Column layout for the Monolith AIR execution trace.
//!
//! # Overview
//!
//! Each row of the trace represents one complete Monolith permutation.
//! The columns are organized to mirror the round structure of the permutation:
//!
//! ```text
//!   ┌────────┬────────────────────────────────────┬────────────────────┐
//!   │ inputs │  full rounds (NUM_FULL_ROUNDS)     │  final round       │
//!   │ [W]    │  [bars_bits | bars_out | post] × R │  [bars_bits | ...] │
//!   └────────┴────────────────────────────────────┴────────────────────┘
//! ```
//!
//! # Column Count
//!
//! The total number of columns per row depends on the parameters:
//!
//! - **Inputs**: `WIDTH` columns.
//! - **Per round** (NUM_FULL_ROUNDS + 1 total rounds):
//!   `NUM_BARS * (2 * FIELD_BITS + NUM_MATCH_FLAGS)` (bit decomposition,
//!   chi products, and match flags)
//!   + `NUM_BARS` (bar outputs)
//!   + `WIDTH` (post-state).
//!
//! # Monolith Round Structure
//!
//! Each round of the Monolith permutation applies three layers:
//!
//! 1. **Bars**: Kintsugi-based S-box on the first `NUM_BARS` state elements.
//!    The field element is decomposed into limbs, a chi-like S-box is applied
//!    per limb, and the results are recomposed. This requires storing the
//!    bit decomposition of each Bar'd element for constraint verification.
//!
//! 2. **Bricks**: Feistel Type-3 layer with squaring:
//!    `state[i] += state[i-1]^2` for `i > 0`. This is a degree-2 operation.
//!
//! 3. **Concrete**: Multiplication by the circulant MDS matrix (linear).

use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

/// Column layout for one complete Monolith permutation (one trace row).
#[repr(C)]
pub struct MonolithCols<
    T,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
> {
    /// The initial permutation state before any operations are applied.
    pub inputs: [T; WIDTH],

    /// Columns for the `NUM_FULL_ROUNDS` rounds that include round constants.
    ///
    /// Each round applies: Bars → Bricks → Concrete → AddRoundConstants.
    /// The `post` field in each round stores the state after all four operations.
    pub full_rounds:
        [MonolithRoundCols<T, WIDTH, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS>; NUM_FULL_ROUNDS],

    /// Columns for the final round (no round constants).
    ///
    /// The final round applies: Bars → Bricks → Concrete.
    /// The `post` field stores the final permutation output.
    pub final_round: MonolithRoundCols<T, WIDTH, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS>,
}

/// Columns for a single Monolith round.
///
/// Each round applies: Bars → Bricks → Concrete (→ AddRC for non-final rounds).
#[repr(C)]
pub struct MonolithRoundCols<
    T,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
> {
    /// Bit decomposition of the input to each Bar application (LSB first).
    ///
    /// For each of the `NUM_BARS` elements entering the Bars layer, we store
    /// `FIELD_BITS` bits. These committed bits are used to:
    ///
    /// 1. Verify the element reconstruction: `sum(bits[i] * 2^i) == state[j]`.
    /// 2. Evaluate the chi S-box algebraically per limb.
    /// 3. Verify the S-box output: `bars_output[j] == Bar(state[j])`.
    ///
    /// Each bit is constrained to be boolean: `b * (1 - b) = 0`.
    pub bars_input_bits: [[T; FIELD_BITS]; NUM_BARS],

    /// Committed chi AND product, one cell per output bit.
    ///
    /// # Binding equation (limb of width n)
    ///
    /// - Width 8: `chi[j] = (1 - x_{j-2}) * x_{j-3} * x_{j-4}` — degree 3.
    /// - Width 7: `chi[j] = (1 - x_{j-2}) * x_{j-3}`           — degree 2.
    ///
    /// All indices are taken mod n.
    ///
    /// # Why this column exists
    ///
    /// - Inlining the chi XOR makes the limb output degree 4.
    /// - Committing the AND product splits it into a degree-3 binding
    ///   and a degree-2 XOR, capping the AIR at degree 3.
    pub bars_chi_products: [[T; FIELD_BITS]; NUM_BARS],

    /// Running "still matches the modulus prefix" flag, one cell per
    /// modulus **one**-bit.
    ///
    /// # Walk (MSB to LSB)
    ///
    /// - Start with `m = 1` above the most significant bit.
    /// - For each bit position i from MSB down to LSB:
    ///   - If the modulus has 1 at i: commit the next flag cell,
    ///     `m_i = m_{i+1} * x_i`.
    ///   - If the modulus has 0 at i: no cell is committed; `m_i = m_{i+1}`
    ///     is enforced only through the side constraint
    ///     `m_{i+1} * x_i = 0` (any 1 in x while still matching means
    ///     `x > p`), with `m_{i+1}` reused directly from the previous step.
    /// - Final: the last committed flag is `0`, ruling out the encoding
    ///   `x == p`.
    ///
    /// A modulus-zero bit is a pure copy of the previous flag, so it needs
    /// no committed column of its own — only `NUM_MATCH_FLAGS` cells (the
    /// modulus's Hamming weight) are stored per Bar, one per one-bit.
    ///
    /// # Why this column exists
    ///
    /// - Without it, integers in `[p, 2^FIELD_BITS - 1]` would forge a second
    ///   bit encoding for field elements already represented canonically.
    /// - The walk works for any prime, including those whose forbidden range
    ///   covers many bit patterns (e.g. Goldilocks).
    pub bars_match_flags: [[T; NUM_MATCH_FLAGS]; NUM_BARS],

    /// Committed output of each Bar (S-box) application.
    ///
    /// These values reset the expression degree to 1 before the Bricks layer,
    /// ensuring that the squaring in Bricks produces degree-2 constraints
    /// rather than allowing degree blowup.
    ///
    /// Elements `NUM_BARS..WIDTH` of the state pass through Bars unchanged.
    pub bars_output: [T; NUM_BARS],

    /// The complete state after Bricks + Concrete (+ round constants for
    /// non-final rounds).
    ///
    /// These committed values reset all expressions to degree 1, preventing
    /// degree blowup across rounds.
    pub post: [T; WIDTH],
}

/// Compute the total number of columns per trace row.
///
/// Uses `size_of::<MonolithCols<u8, ...>>()` since `u8` has size 1.
/// Each `u8` field maps to exactly one column.
pub const fn num_cols<
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
>() -> usize {
    size_of::<MonolithCols<u8, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS>>()
}

impl<
    T,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
> Borrow<MonolithCols<T, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS>> for [T]
{
    fn borrow(
        &self,
    ) -> &MonolithCols<T, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS> {
        // Reinterpret the flat slice as the column struct.
        // Safety: `#[repr(C)]` guarantees predictable layout and `T` has
        // alignment 1 for the `u8` case used in `num_cols`.
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<MonolithCols<T, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS>>()
        };

        // No padding before or after the struct.
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");

        // The slice must contain exactly one struct.
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<
    T,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
    const NUM_MATCH_FLAGS: usize,
> BorrowMut<MonolithCols<T, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS>>
    for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut MonolithCols<T, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS> {
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<MonolithCols<T, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS, NUM_MATCH_FLAGS>>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
