//! Column layout for the Poseidon1 AIR execution trace.
//!
//! # Overview
//!
//! Each row of the trace represents one complete Poseidon1 permutation.
//! The columns are organized to mirror the three phases of the permutation:
//!
//! ```text
//!   ┌────────┬──────────────────────────┬──────────────────┬──────────────────────────┐
//!   │ inputs │  beginning full rounds   │  partial rounds  │  ending full rounds      │
//!   │ [W]    │  [HALF_FULL_ROUNDS]      │  [PARTIAL_ROUNDS]│  [HALF_FULL_ROUNDS]      │
//!   └────────┴──────────────────────────┴──────────────────┴──────────────────────────┘
//! ```
//!
//! # Column Count
//!
//! The total number of columns per row depends on the parameters:
//!
//! - **Inputs**: `WIDTH` columns.
//! - **Full rounds** (beginning + ending): `2 * HALF_FULL_ROUNDS * (WIDTH * (1 + SBOX_REGISTERS) + WIDTH)`.
//! - **Partial rounds**: `PARTIAL_ROUNDS * (SBOX_REGISTERS + WIDTH)`.

use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

/// Column layout for one Poseidon1 permutation (one trace row).
///
/// # Type Parameters
///
/// - `T`: The element type. Either:
///   - a concrete field element (during trace generation),
///   - an `AirBuilder::Var` (during constraint evaluation).
/// - `WIDTH`: Number of field elements in the permutation state (`t` in the paper).
/// - `SBOX_DEGREE`: The S-box exponent `α` (e.g., 3, 5, 7, or 11).
/// - `SBOX_REGISTERS`: Number of intermediate values stored per S-box evaluation.
///   Depends on `SBOX_DEGREE` (e.g., 0 for degree 3, 1 for degree 7).
/// - `HALF_FULL_ROUNDS`: Number of full rounds in each half (`RF/2`).
/// - `PARTIAL_ROUNDS`: Number of partial rounds (`RP`).
///
/// # Layout
///
/// ```text
///   PoseidonCols
///   ├── inputs:                [T; WIDTH]
///   ├── beginning_full_rounds: [FullRound; HALF_FULL_ROUNDS]
///   ├── partial_rounds:        [PartialRound; PARTIAL_ROUNDS]
///   └── ending_full_rounds:    [FullRound; HALF_FULL_ROUNDS]
/// ```
#[repr(C)]
pub struct PoseidonCols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    /// The initial permutation state before any rounds are applied.
    pub inputs: [T; WIDTH],

    /// Columns for the first `RF/2` full rounds.
    ///
    /// These rounds apply the S-box to every state element, providing strong
    /// resistance against statistical attacks (see Poseidon paper, Section 5.5.1).
    pub beginning_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],

    /// Columns for the `RP` partial rounds.
    ///
    /// These rounds apply the S-box only to `state[0]`, which is cheaper but
    /// still increases the algebraic degree to resist algebraic attacks
    /// (see Poseidon paper, Section 5.5.2).
    pub partial_rounds: [PartialRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; PARTIAL_ROUNDS],

    /// Columns for the last `RF/2` full rounds.
    pub ending_full_rounds: [FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>; HALF_FULL_ROUNDS],
}

/// Columns for one full round of the Poseidon1 permutation.
///
/// In a full round, every state element passes through the S-box:
///
/// ```text
///   ┌──────────────────────────────────────────────────────────────────┐
///   │  state[0]  state[1]  state[2]  ...  state[W-1]                   │
///   │     │         │         │               │                        │
///   │     ▼         ▼         ▼               ▼                        │
///   │   S-box     S-box     S-box     ...   S-box     (+ round consts) │
///   │     │         │         │               │                        │
///   │     └─────────┴─────────┴───────────────┘                        │
///   │                       ▼                                          │
///   │                  MDS multiply                                    │
///   │                       │                                          │
///   │                       ▼                                          │
///   │                   post[0..W]                                     │
///   └──────────────────────────────────────────────────────────────────┘
/// ```
#[repr(C)]
pub struct FullRound<T, const WIDTH: usize, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize> {
    /// S-box intermediate values for each of the `WIDTH` state elements.
    ///
    /// Each `SBox` stores the intermediate powers needed to verify the S-box
    /// computation (e.g., `x^3` for a degree-7 S-box).
    pub sbox: [SBox<T, SBOX_DEGREE, SBOX_REGISTERS>; WIDTH],

    /// The complete state after the MDS multiply.
    ///
    /// These committed values reset the expression degree to 1, preventing
    /// degree blowup across rounds.
    pub post: [T; WIDTH],
}

/// Columns for one partial round of the Poseidon1 permutation.
///
/// In a partial round, only `state[0]` passes through the S-box:
///
/// ```text
///   ┌──────────────────────────────────────────────────────────────────┐
///   │  state[0]  state[1]  state[2]  ...  state[W-1]                   │
///   │     │         │         │               │                        │
///   │     ▼         │         │               │                        │
///   │  S-box      (identity) (identity) ... (identity) (+ round const) │
///   │     │         │         │               │                        │
///   │     └─────────┴─────────┴───────────────┘                        │
///   │                       ▼                                          │
///   │              MDS multiply (dense)                                │
///   │                       │                                          │
///   │                       ▼                                          │
///   │                   post[0..W]                                     │
///   └──────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Why store the full post-state?
///
/// The dense MDS matrix in Poseidon1 mixes **all** elements every round.
/// If we only stored `post[0]` (as one might with a sparse internal matrix),
/// the AIR expressions for subsequent rounds would accumulate multiplicative
/// degree from every MDS entry, causing exponential expression blowup.
///
/// Storing all `WIDTH` post-state values resets the expression degree to 1.
#[repr(C)]
pub struct PartialRound<T, const WIDTH: usize, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize>
{
    /// S-box intermediate values for `state[0]` only.
    pub sbox: SBox<T, SBOX_DEGREE, SBOX_REGISTERS>,

    /// The complete state after the dense MDS multiply.
    pub post: [T; WIDTH],
}

/// Intermediate values for one S-box evaluation (`x → x^DEGREE`).
///
/// The number of stored intermediates depends on the degree:
///
/// | `DEGREE` | `REGISTERS` | Intermediates | Computation               |
/// |----------|-------------|---------------|---------------------------|
/// | 3        | 0           | (none)        | `x^3` directly            |
/// | 5        | 0           | (none)        | `x^5` directly            |
/// | 7        | 0           | (none)        | `x^7` directly            |
/// | 5        | 1           | `x^3`         | `x^3 * x^2`               |
/// | 7        | 1           | `x^3`         | `(x^3)^2 * x`             |
/// | 11       | 2           | `x^3`, `x^9`  | `x^9 * x^2`               |
///
/// When `REGISTERS = 0`, the S-box output is computed directly without
/// intermediate columns. This trades constraint degree for column count.
///
/// When `REGISTERS > 0`, intermediate values are committed to the trace,
/// keeping the constraint degree lower at the cost of more columns.
#[repr(C)]
pub struct SBox<T, const DEGREE: u64, const REGISTERS: usize>(pub [T; REGISTERS]);

/// Compute the total number of columns per trace row.
///
/// Uses `size_of::<PoseidonCols<u8, ...>>()` since `u8` has size 1.
/// Each `u8` field maps to exactly one column.
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
        // Reinterpret the flat slice as the column struct.
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
