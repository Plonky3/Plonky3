use alloc::vec::Vec;
use core::array;

use p3_air::utils::u32_to_bits_le;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::columns::{Blake3Cols, NUM_BLAKE3_COLS};
use crate::constants::{IV, permute};
use crate::{Blake3CompressionInput, Blake3State, FullRound};

/// Generate a prime-field Blake-3 trace using legacy benchmark metadata.
///
/// Each input contains 16 message words followed by eight chaining words. The counter is the
/// row index, the block length is the batch size cast to `u32`, and flags are zero. Use
/// [`generate_trace_rows_with_inputs`] to supply explicit compression metadata.
// TODO: Take generic iterable
pub fn generate_trace_rows<F: PrimeField64>(
    inputs: Vec<[u32; 24]>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let block_len = inputs.len() as u32;
    let inputs = inputs
        .into_iter()
        .enumerate()
        .map(|(counter, input)| Blake3CompressionInput {
            chaining_value: array::from_fn(|i| input[16 + i]),
            block: array::from_fn(|i| input[i]),
            counter: counter as u64,
            block_len,
            flags: 0,
        })
        .collect();
    generate_trace_rows_with_inputs(inputs, extra_capacity_bits)
}

/// Generate a prime-field Blake-3 trace with one independent compression per row.
///
/// Metadata comes entirely from each input and is independent of its position or batch size.
/// This is the raw compression function: callers supply padded message words, chaining values,
/// counters, lengths and flags. It does not perform message hashing or bind inputs or outputs
/// to AIR public values.
///
/// # Panics
///
/// Panics if `inputs` is empty or its length is not a power of two.
#[instrument(name = "generate Blake3 trace", skip_all)]
pub fn generate_trace_rows_with_inputs<F: PrimeField64>(
    inputs: Vec<Blake3CompressionInput>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let num_rows = inputs.len();
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
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
        .for_each(|(row, input)| generate_trace_rows_for_perm(row, input));

    trace
}

/// Each row is one Blake-3 compression.
fn generate_trace_rows_for_perm<F: PrimeField64>(
    row: &mut Blake3Cols<F>,
    input: Blake3CompressionInput,
) {
    row.inputs = input.block.map(u32_to_bits_le);
    row.chaining_values =
        array::from_fn(|i| array::from_fn(|j| u32_to_bits_le(input.chaining_value[4 * i + j])));

    let counter_low = input.counter as u32;
    let counter_hi = (input.counter >> 32) as u32;
    row.counter_low = u32_to_bits_le(counter_low);
    row.counter_hi = u32_to_bits_le(counter_hi);
    row.block_len = u32_to_bits_le(input.block_len);
    row.flags = u32_to_bits_le(input.flags);

    row.initial_row0 = array::from_fn(|i| {
        [
            F::from_u16(input.chaining_value[i] as u16),
            F::from_u16((input.chaining_value[i] >> 16) as u16),
        ]
    });

    row.initial_row2 = array::from_fn(|i| [F::from_u16(IV[i][0]), F::from_u16(IV[i][1])]);

    // Compute on u32 words while recording the intermediate states in the trace.
    let mut m_vec = input.block;
    let mut state = [
        array::from_fn(|i| input.chaining_value[i]),
        array::from_fn(|i| input.chaining_value[4 + i]),
        [
            (IV[0][0] as u32) + ((IV[0][1] as u32) << 16),
            (IV[1][0] as u32) + ((IV[1][1] as u32) << 16),
            (IV[2][0] as u32) + ((IV[2][1] as u32) << 16),
            (IV[3][0] as u32) + ((IV[3][1] as u32) << 16),
        ],
        [counter_low, counter_hi, input.block_len, input.flags],
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
    row.outputs[2] = array::from_fn(|i| u32_to_bits_le(state[2][i] ^ input.chaining_value[i]));
    row.outputs[3] = array::from_fn(|i| u32_to_bits_le(state[3][i] ^ input.chaining_value[4 + i]));
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
    use alloc::vec::Vec;
    use core::array;
    use core::borrow::{Borrow, BorrowMut};

    use blake3::platform::Platform;
    use p3_air::utils::u32_to_bits_le;
    use p3_air::{check_all_constraints, check_constraints};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;

    use super::{
        generate_trace_rows, generate_trace_rows_for_perm, generate_trace_rows_with_inputs,
    };
    use crate::{Blake3Air, Blake3Cols, Blake3CompressionInput, NUM_BLAKE3_COLS};

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
            assert_eq!(row.block_len, u32_to_bits_le(8));
            assert_eq!(row.flags, [BabyBear::ZERO; 32]);
            let expected = Platform::Portable.compress_xof(&[0; 8], &[0; 64], 8, counter as u64, 0);
            for (bits, bytes) in row
                .outputs
                .iter()
                .flatten()
                .zip(expected.as_chunks::<4>().0)
            {
                assert_eq!(*bits, u32_to_bits_le(u32::from_le_bytes(*bytes)));
            }
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
            generate_trace_rows_for_perm(
                row,
                Blake3CompressionInput {
                    chaining_value,
                    block: array::from_fn(|i| input[i]),
                    counter: row_index as u64,
                    block_len: 64,
                    flags: 0,
                },
            );

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

    fn compression_inputs() -> Vec<Blake3CompressionInput> {
        let counters = [
            0,
            1,
            u32::MAX as u64,
            1 << 32,
            0x1234_5678_9abc_def0,
            u64::MAX,
            7,
            0,
        ];
        let lengths = [0, 1, 4, 17, 32, 55, 63, 64];
        let flags = [11, 1, 2, 4, 8, 16, 32, 64];
        (0..8)
            .map(|i| Blake3CompressionInput {
                chaining_value: array::from_fn(|j| (j as u32).wrapping_mul(0x9e37_79b9)),
                block: array::from_fn(|j| {
                    u32::from_le_bytes(array::from_fn(|k| {
                        let byte = 4 * j + k;
                        if byte < lengths[i] as usize {
                            byte as u8
                        } else {
                            0
                        }
                    }))
                }),
                counter: counters[i],
                block_len: lengths[i],
                flags: flags[i],
            })
            .collect()
    }

    #[test]
    fn explicit_metadata_matches_upstream_compression_and_batch_regrouping() {
        let inputs = compression_inputs();
        let trace = generate_trace_rows_with_inputs::<BabyBear>(inputs.clone(), 1);
        check_constraints(&Blake3Air {}, &trace, &[]);
        for (i, input) in inputs.iter().enumerate() {
            let row = trace.row_slice(i).unwrap();
            let row: &Blake3Cols<BabyBear> = (*row).borrow();
            let block = array::from_fn(|j| input.block[j / 4].to_le_bytes()[j % 4]);
            let expected = Platform::Portable.compress_xof(
                &input.chaining_value,
                &block,
                input.block_len as u8,
                input.counter,
                input.flags as u8,
            );
            for (word, bytes) in row
                .outputs
                .iter()
                .flatten()
                .zip(expected.as_chunks::<4>().0)
            {
                assert_eq!(
                    *word,
                    u32_to_bits_le(u32::from_le_bytes(*bytes)),
                    "input {i}"
                );
            }
            let alone = generate_trace_rows_with_inputs::<BabyBear>(vec![*input], 0);
            assert_eq!(&*trace.row_slice(i).unwrap(), &*alone.row_slice(0).unwrap());
        }
        let mut reordered = inputs.clone();
        reordered.reverse();
        let reordered = generate_trace_rows_with_inputs::<BabyBear>(reordered, 0);
        for i in 0..inputs.len() {
            assert_eq!(
                &*trace.row_slice(i).unwrap(),
                &*reordered.row_slice(inputs.len() - 1 - i).unwrap()
            );
        }
    }

    #[test]
    fn explicit_metadata_and_output_tampering_is_rejected() {
        for field in 0..20 {
            let mut trace = generate_trace_rows_with_inputs::<BabyBear>(compression_inputs(), 0);
            let row: &mut Blake3Cols<BabyBear> = trace.row_mut(3).borrow_mut();
            let bit = match field {
                0 => &mut row.counter_low[0],
                1 => &mut row.counter_hi[31],
                2 => &mut row.block_len[0],
                3 => &mut row.flags[0],
                output => &mut row.outputs[(output - 4) / 4][(output - 4) % 4][31],
            };
            *bit = BabyBear::ONE - *bit;
            let failures = check_all_constraints(&Blake3Air {}, &trace, &[], None).failures;
            assert!(!failures.is_empty(), "field {field}");
            assert!(failures.iter().all(|failure| failure.row == 3));
        }
    }

    #[test]
    fn explicit_metadata_trace_proves_and_verifies() {
        use p3_baby_bear::Poseidon2BabyBear;
        use p3_challenger::DuplexChallenger;
        use p3_commit::ExtensionMmcs;
        use p3_dft::Radix2DitParallel;
        use p3_field::Field;
        use p3_field::extension::BinomialExtensionField;
        use p3_fri::{FriParameters, TwoAdicFriPcs};
        use p3_merkle_tree::MerkleTreeMmcs;
        use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
        use p3_uni_stark::{StarkConfig, prove, verify};
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        type Perm = Poseidon2BabyBear<16>;
        type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
        type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
        type ValMmcs = MerkleTreeMmcs<
            <BabyBear as Field>::Packing,
            <BabyBear as Field>::Packing,
            Hash,
            Compress,
            2,
            8,
        >;
        type Challenge = BinomialExtensionField<BabyBear, 4>;
        type ChallengeMmcs = ExtensionMmcs<BabyBear, Challenge, ValMmcs>;
        type Pcs = TwoAdicFriPcs<BabyBear, Radix2DitParallel<BabyBear>, ValMmcs, ChallengeMmcs>;
        type Challenger = DuplexChallenger<BabyBear, Perm, 16, 8>;
        type Config = StarkConfig<Pcs, Challenge, Challenger>;

        let mut rng = SmallRng::seed_from_u64(17);
        let perm = Perm::new_from_rng_128(&mut rng);
        let val_mmcs = ValMmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
        // Small proof parameters are only for this test.
        let fri_params = FriParameters {
            log_blowup: 2,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: ChallengeMmcs::new(val_mmcs.clone()),
        };
        let pcs = Pcs::new(Radix2DitParallel::default(), val_mmcs, fri_params);
        let config = Config::new(pcs, Challenger::new(perm));
        let trace = generate_trace_rows_with_inputs::<BabyBear>(compression_inputs(), 0);
        let mut proof = prove(&config, &Blake3Air {}, trace, &[]).unwrap();
        verify(&config, &Blake3Air {}, &proof, &[]).unwrap();
        proof.opened_values.trace_local[0] += Challenge::ONE;
        assert!(verify(&config, &Blake3Air {}, &proof, &[]).is_err());
    }
}
