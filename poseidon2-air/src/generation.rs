use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::columns::{num_cols, Poseidon2Cols};

// TODO: Take generic iterable
#[instrument(name = "generate Poseidon2 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );
    let mut trace = RowMajorMatrix::new(
        vec![F::zero(); n],
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(),
    );
    let (prefix, rows, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows.par_iter_mut().zip(inputs).for_each(|(row, input)| {
        generate_trace_rows_for_perm(row, input);
    });

    trace
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
fn generate_trace_rows_for_perm<
    F: PrimeField,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    row: &mut Poseidon2Cols<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    input: [F; WIDTH],
) {
}
