use core::array;
use core::borrow::Borrow;

use p3_air::AirBuilder;
use p3_matrix::Matrix;

use crate::columns::KeccakCols;
use crate::{NUM_ROUNDS, NUM_ROUNDS_MIN_1};

/// Evaluate and constrain round flags for each row of the Keccak AIR.
///
/// # Overview
///
/// - Enforces that in the first row, `step_flags[0]` is 1, and all other flags are 0.
/// - Enforces that at each transition, the flags rotate forward (circular shift).
/// - Guarantees that exactly one round flag is active per row, following Keccak's round schedule.
///
/// # Arguments
///
/// - `builder`: An `AirBuilder` used to express constraints on the AIR trace.
#[inline]
pub(crate) fn eval_round_flags<AB: AirBuilder>(builder: &mut AB) {
    // Access the main trace matrix.
    let main = builder.main();

    // Get the local (current) row and the next row slices.
    let (local, next) = (
        main.row_slice(0).expect("The matrix is empty?"),
        main.row_slice(1).expect("The matrix only has 1 row?"),
    );

    // Cast slices into typed Keccak column references.
    let local: &KeccakCols<AB::Var> = (*local).borrow();
    let next: &KeccakCols<AB::Var> = (*next).borrow();

    // Initially, the first step flag should be 1 while the others should be 0.
    //
    // Constraint: In the first row, the first flag is 1.
    builder
        .when_first_row()
        .assert_one(local.step_flags[0].clone());
    // Constraint: In the first row, all other flags are 0.
    builder
        .when_first_row()
        .assert_zeros::<NUM_ROUNDS_MIN_1, _>(try_clone_array(&local.step_flags[1..]));

    // Constraint: In all transitions, flags rotate forward.
    //
    // Formally, for each flag i in the local row, it should equal the next row's flag at (i + 1) mod NUM_ROUNDS.
    //
    // This ensures that exactly one flag "moves forward" each step in a cyclic manner.
    builder
        .when_transition()
        .assert_zeros::<NUM_ROUNDS, _>(array::from_fn(|i| {
            local.step_flags[i].clone() - next.step_flags[(i + 1) % NUM_ROUNDS].clone()
        }));
}

/// Clone a slice into an array of fixed length N by element-wise cloning.
///
/// # Panics
///
/// Panics if the input slice length does not match N.
///
/// # Arguments
///
/// - `slice`: The input slice to copy.
///
/// # Returns
///
/// - `[T; N]`: The cloned array.
fn try_clone_array<T: Clone, const N: usize>(slice: &[T]) -> [T; N] {
    // Check at runtime that the length is correct (should always hold).
    assert!(slice.len() == N, "Incorrect length");

    // Clone each element into a new array.
    array::from_fn(|i| slice[i].clone())
}
