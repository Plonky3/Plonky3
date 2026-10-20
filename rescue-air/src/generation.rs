use alloc::vec;
use core::borrow::BorrowMut;

use p3_field::{Field, PermutationMonomial};
use p3_matrix::dense::RowMajorMatrix;

use crate::columns::{RescueCols, num_cols};

/// Generate a trace with one Rescue permutation per row.
///
/// The rows replay the permutation exactly as [`p3_rescue::Rescue`] does, so a
/// trace accepted by [`crate::RescueAir`] is a trace of that permutation.
/// The number of rows is padded to a power of two with all-zero inputs.
///
/// # Panics
/// If `round_constants` is not `2 * WIDTH * NUM_ROUNDS` long.
pub fn generate_trace_rows<F, const WIDTH: usize, const ALPHA: u64, const NUM_ROUNDS: usize>(
    inputs: &[[F; WIDTH]],
    mds: &[[F; WIDTH]; WIDTH],
    round_constants: &[F],
) -> RowMajorMatrix<F>
where
    F: Field + PermutationMonomial<ALPHA>,
{
    assert_eq!(
        round_constants.len(),
        2 * WIDTH * NUM_ROUNDS,
        "round_constants must have length 2 * WIDTH * NUM_ROUNDS"
    );

    let width = num_cols::<WIDTH, NUM_ROUNDS>();
    let num_rows = inputs.len().next_power_of_two().max(1);
    let mut values = vec![F::ZERO; num_rows * width];

    for (row, input) in values.chunks_exact_mut(width).zip(
        inputs
            .iter()
            .copied()
            .chain(core::iter::repeat([F::ZERO; WIDTH])),
    ) {
        generate_trace_row::<F, WIDTH, ALPHA, NUM_ROUNDS>(row, input, mds, round_constants);
    }

    RowMajorMatrix::new(values, width)
}

fn apply_mds<F: Field, const WIDTH: usize>(
    mds: &[[F; WIDTH]; WIDTH],
    state: &[F; WIDTH],
) -> [F; WIDTH] {
    core::array::from_fn(|i| {
        (0..WIDTH)
            .map(|j| mds[i][j] * state[j])
            .fold(F::ZERO, |a, b| a + b)
    })
}

fn generate_trace_row<F, const WIDTH: usize, const ALPHA: u64, const NUM_ROUNDS: usize>(
    row: &mut [F],
    input: [F; WIDTH],
    mds: &[[F; WIDTH]; WIDTH],
    round_constants: &[F],
) where
    F: Field + PermutationMonomial<ALPHA>,
{
    let cols: &mut RescueCols<F, WIDTH, NUM_ROUNDS> = row.borrow_mut();
    cols.inputs = input;

    let mut state = input;
    for r in 0..NUM_ROUNDS {
        // Forward S-box.
        let sbox_pow: [F; WIDTH] = core::array::from_fn(|i| state[i].injective_exp_n());
        cols.rounds[r].sbox_pow = sbox_pow;

        // MDS, then this half-round's constants.
        let mut mid = apply_mds(mds, &sbox_pow);
        for i in 0..WIDTH {
            mid[i] += round_constants[r * 2 * WIDTH + i];
        }
        cols.rounds[r].mid = mid;

        // Inverse S-box.
        let inv_sbox: [F; WIDTH] = core::array::from_fn(|i| mid[i].injective_exp_root_n());
        cols.rounds[r].inv_sbox = inv_sbox;

        // MDS, then this half-round's constants.
        let mut post = apply_mds(mds, &inv_sbox);
        for i in 0..WIDTH {
            post[i] += round_constants[r * 2 * WIDTH + WIDTH + i];
        }
        cols.rounds[r].post = post;

        state = post;
    }
}
