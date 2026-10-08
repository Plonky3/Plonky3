use alloc::vec::Vec;
use core::array;

use p3_air::utils::u32_to_bits_le;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::columns::{Blake3Cols, NUM_BLAKE3_COLS};
use crate::constants::{IV, permute};
use crate::{Blake3State, FullRound};

// TODO: Take generic iterable
#[instrument(name = "generate Blake3 trace", skip_all)]
pub fn generate_trace_rows<F: PrimeField64>(
    inputs: Vec<[u32; 24]>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let num_rows = inputs.len();
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to VECTOR_LEN times a power of two"
    );

    let trace_length = num_rows * NUM_BLAKE3_COLS;

    // We allocate extra_capacity_bits now as this will be needed by the dft.
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    long_trace.truncate(trace_length);

    let mut trace = RowMajorMatrix::new(long_trace, NUM_BLAKE3_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Blake3Cols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    rows.par_iter_mut()
        .zip(inputs)
        .enumerate()
        .for_each(|(counter, (row, input))| {
            generate_trace_rows_for_perm(row, input, counter, num_rows);
        });

    trace
}

/// Each row is one full implementation of the Blake-3 hash.
fn generate_trace_rows_for_perm<F: PrimeField64>(
    row: &mut Blake3Cols<F>,
    input: [u32; 24],
    counter: usize,
    block_len: usize,
) {
    // We split the input into 2 parts.
    // The first 16 elements we treat as the inputs or block_words
    row.inputs = array::from_fn(|i| u32_to_bits_le(input[i]));

    // the remaining 8 elements are interpreted as the chaining values.
    row.chaining_values =
        array::from_fn(|i| array::from_fn(|j| u32_to_bits_le(input[16 + 4 * i + j])));

    // Widen before shifting so the high word is zero on 32-bit targets.
    let counter = counter as u64;
    let counter_low = counter as u32;
    let counter_hi = (counter >> 32) as u32;

    // Use the same words in the counter columns and the compression state.
    row.counter_low = u32_to_bits_le(counter_low);
    row.counter_hi = u32_to_bits_le(counter_hi);
    row.block_len = u32_to_bits_le(block_len as u32);

    row.initial_row0 = array::from_fn(|i| {
        [
            F::from_u16(input[16 + i] as u16),
            F::from_u16((input[16 + i] >> 16) as u16),
        ]
    });

    row.initial_row2 = array::from_fn(|i| [F::from_u16(IV[i][0]), F::from_u16(IV[i][1])]);

    // We save the state and m_vec as u_32's we will quickly compute the hash using these whilst saving
    // the appropriate data in the trace as we go.
    let mut m_vec: [u32; 16] = array::from_fn(|i| input[i]);
    let mut state = [
        [input[16], input[16 + 1], input[16 + 2], input[16 + 3]],
        [input[16 + 4], input[16 + 5], input[16 + 6], input[16 + 7]],
        [
            (IV[0][0] as u32) + ((IV[0][1] as u32) << 16),
            (IV[1][0] as u32) + ((IV[1][1] as u32) << 16),
            (IV[2][0] as u32) + ((IV[2][1] as u32) << 16),
            (IV[3][0] as u32) + ((IV[3][1] as u32) << 16),
        ],
        [counter_low, counter_hi, block_len as u32, 0],
    ];

    generate_trace_row_for_round(&mut row.full_rounds[0], &mut state, &m_vec); // round 1
    permute(&mut m_vec);
    generate_trace_row_for_round(&mut row.full_rounds[1], &mut state, &m_vec); // round 2
    permute(&mut m_vec);
    generate_trace_row_for_round(&mut row.full_rounds[2], &mut state, &m_vec); // round 3
    permute(&mut m_vec);
    generate_trace_row_for_round(&mut row.full_rounds[3], &mut state, &m_vec); // round 4
    permute(&mut m_vec);
    generate_trace_row_for_round(&mut row.full_rounds[4], &mut state, &m_vec); // round 5
    permute(&mut m_vec);
    generate_trace_row_for_round(&mut row.full_rounds[5], &mut state, &m_vec); // round 6
    permute(&mut m_vec);
    generate_trace_row_for_round(&mut row.full_rounds[6], &mut state, &m_vec); // round 7

    // After performing all the rounds, all that is left to do is to populate the final xor data.

    row.final_round_helpers = array::from_fn(|i| u32_to_bits_le(state[2][i]));

    row.outputs[0] = array::from_fn(|i| u32_to_bits_le(state[0][i] ^ state[2][i]));
    row.outputs[1] = array::from_fn(|i| u32_to_bits_le(state[1][i] ^ state[3][i]));
    row.outputs[2] = array::from_fn(|i| u32_to_bits_le(state[2][i] ^ input[16 + i]));
    row.outputs[3] = array::from_fn(|i| u32_to_bits_le(state[3][i] ^ input[20 + i]));
}

fn generate_trace_row_for_round<F: PrimeField64>(
    round_data: &mut FullRound<F>,
    state: &mut [[u32; 4]; 4],
    m_vec: &[u32; 16],
) {
    // We populate the round_data as we iterate through and compute the permutation following the reference implementation.

    // We start by performing the first half of the four column quarter round functions.
    (0..4).for_each(|i| {
        (state[0][i], state[1][i], state[2][i], state[3][i]) = verifiable_half_round(
            state[0][i],
            state[1][i],
            state[2][i],
            state[3][i],
            m_vec[2 * i],
            false,
        );
    });

    // After the first four operations we need to save a copy of the state into the trace.
    save_state_to_trace(&mut round_data.state_prime, state);

    // Next we do the second half of the four column quarter round functions.
    (0..4).for_each(|i| {
        (state[0][i], state[1][i], state[2][i], state[3][i]) = verifiable_half_round(
            state[0][i],
            state[1][i],
            state[2][i],
            state[3][i],
            m_vec[2 * i + 1],
            true,
        );
    });

    // Again we save another copy of the state.
    save_state_to_trace(&mut round_data.state_middle, state);

    // We repeat with the diagonals quarter round function.

    // Do the first half of the four diagonal quarter round functions.
    (0..4).for_each(|i| {
        (
            state[0][i],
            state[1][(i + 1) % 4],
            state[2][(i + 2) % 4],
            state[3][(i + 3) % 4],
        ) = verifiable_half_round(
            state[0][i],
            state[1][(i + 1) % 4],
            state[2][(i + 2) % 4],
            state[3][(i + 3) % 4],
            m_vec[8 + 2 * i],
            false,
        );
    });

    // Save a copy of the state to the trace.
    save_state_to_trace(&mut round_data.state_middle_prime, state);

    // Do the second half of the four diagonal quarter round functions.
    (0..4).for_each(|i| {
        (
            state[0][i],
            state[1][(i + 1) % 4],
            state[2][(i + 2) % 4],
            state[3][(i + 3) % 4],
        ) = verifiable_half_round(
            state[0][i],
            state[1][(i + 1) % 4],
            state[2][(i + 2) % 4],
            state[3][(i + 3) % 4],
            m_vec[9 + 2 * i],
            true,
        );
    });

    // Save a copy of the state to the trace.
    save_state_to_trace(&mut round_data.state_output, state);
}

/// Perform half of a quarter round on the given elements.
///
/// The boolean flag, indicates whether this is the first (false) or second (true) half round.
const fn verifiable_half_round(
    mut a: u32,
    mut b: u32,
    mut c: u32,
    mut d: u32,
    m: u32,
    flag: bool,
) -> (u32, u32, u32, u32) {
    let (rot_1, rot_2) = if flag { (8, 7) } else { (16, 12) };

    // The first summation:
    a = a.wrapping_add(b);
    a = a.wrapping_add(m);

    // The first xor:
    d = (d ^ a).rotate_right(rot_1);

    // The second summation:
    c = c.wrapping_add(d);

    // The second xor:
    b = (b ^ c).rotate_right(rot_2);

    (a, b, c, d)
}

fn save_state_to_trace<R: PrimeCharacteristicRing>(
    trace: &mut Blake3State<R>,
    state: &[[u32; 4]; 4],
) {
    trace.row0 = array::from_fn(|i| {
        [
            R::from_u16(state[0][i] as u16), // Store the bottom 16 bits packed.
            R::from_u16((state[0][i] >> 16) as u16), // Store the top 16 bits packed.
        ]
    });
    trace.row1 = array::from_fn(|i| u32_to_bits_le(state[1][i])); // Store all 32 bits unpacked.
    trace.row2 = array::from_fn(|i| {
        [
            R::from_u16(state[2][i] as u16), // Store the bottom 16 bits packed.
            R::from_u16((state[2][i] >> 16) as u16), // Store the top 16 bits packed.
        ]
    });
    trace.row3 = array::from_fn(|i| u32_to_bits_le(state[3][i])); // Store all 32 bits unpacked.
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::array;
    use core::borrow::{Borrow, BorrowMut};

    use blake3::platform::Platform;
    use p3_air::check_constraints;
    use p3_air::utils::u32_to_bits_le;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;

    use super::{generate_trace_rows, generate_trace_rows_for_perm};
    use crate::{Blake3Air, Blake3Cols, NUM_BLAKE3_COLS};

    #[test]
    fn trace_counter_words_match_row_indices() {
        // Fixture: eight rows exercise both zero and nonzero row indices.
        let trace = generate_trace_rows::<BabyBear>(vec![[0; 24]; 8], 0);

        // The AIR ties the counter columns to the first compression round.
        check_constraints(&Blake3Air {}, &trace, &[]);

        for counter in 0..trace.height() {
            // Row indices below 2^32 have no high counter bits.
            let row = trace.row_slice(counter).unwrap();
            let row: &Blake3Cols<BabyBear> = (*row).borrow();
            assert_eq!(row.counter_low, u32_to_bits_le(counter as u32));
            assert_eq!(row.counter_hi, [BabyBear::ZERO; 32]);
        }
    }

    #[test]
    fn trace_counter_boundaries_match_reference_compression() {
        // Distinct message and chaining words exercise every output word.
        let input: [u32; 24] = array::from_fn(|i| (i as u32).wrapping_mul(0x9e37_79b9));
        let chaining_value = array::from_fn(|i| input[16 + i]);
        let block = array::from_fn(|i| input[i / 4].to_le_bytes()[i % 4]);

        // Reuse one row instead of allocating a trace with billions of rows.
        let mut trace = generate_trace_rows::<BabyBear>(vec![input], 0);
        for counter in [
            0,
            1,
            5,
            0xffff_ffff,
            0x1_0000_0000,
            0x1234_5678_9abc_def0,
            u64::MAX,
        ] {
            // A row index must fit the target's pointer width.
            let Ok(row_index) = usize::try_from(counter) else {
                continue;
            };

            // Generate the full witness at each counter boundary.
            let row: &mut Blake3Cols<BabyBear> = trace.values[..NUM_BLAKE3_COLS].borrow_mut();
            generate_trace_rows_for_perm(row, input, row_index, 64);

            // Decode the expected words from bytes independently of the shift.
            let bytes = counter.to_le_bytes();
            let low = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            let high = u32::from_le_bytes(bytes[4..].try_into().unwrap());
            assert_eq!(row.counter_low, u32_to_bits_le(low));
            assert_eq!(row.counter_hi, u32_to_bits_le(high));

            // Compare all 16 output words with upstream BLAKE3 compression.
            let expected = Platform::Portable.compress_xof(&chaining_value, &block, 64, counter, 0);
            for (bits, bytes) in row
                .outputs
                .iter()
                .flatten()
                .zip(expected.as_chunks::<4>().0)
            {
                let word = u32::from_le_bytes(*bytes);
                assert_eq!(*bits, u32_to_bits_le(word), "counter {counter:#x}");
            }

            // Counter columns and round witnesses must satisfy the AIR together.
            check_constraints(&Blake3Air {}, &trace, &[]);
        }
    }
}
