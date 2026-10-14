use core::borrow::Borrow;

use p3_baby_bear::BabyBear;
use p3_binary_field::BinaryField128;
use p3_blake3_air::{
    Blake3BinaryAir, Blake3BinaryCols, Blake3BinaryCompactAir, Blake3BinaryCompactCols,
    Blake3CompressionInput, NUM_BLAKE3_BINARY_COMPACT_COLS, generate_binary_compact_trace_packed,
    generate_binary_compact_trace_rows,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;

#[test]
fn compact_packed_bits_match_dense_across_group_and_word_boundaries() {
    let mut rng = SmallRng::seed_from_u64(913);
    for count in [1, 2, 4, 8, 16] {
        let random: Vec<_> = (0..count)
            .map(|_| Blake3CompressionInput {
                chaining_value: rng.random(),
                block: rng.random(),
                counter: rng.random(),
                block_len: rng.random(),
                flags: rng.random(),
            })
            .collect();
        let ones = Blake3CompressionInput {
            chaining_value: [u32::MAX; 8],
            block: [u32::MAX; 16],
            counter: u64::MAX,
            block_len: u32::MAX,
            flags: u32::MAX,
        };
        for inputs in [random, vec![ones; count]] {
            let dense = generate_binary_compact_trace_rows::<F>(inputs.clone(), 0);
            let packed = generate_binary_compact_trace_packed::<F>(inputs);
            assert_eq!(packed.width, NUM_BLAKE3_BINARY_COMPACT_COLS);
            assert_eq!(packed.height(), (count * 8).div_ceil(64));
            for row in 0..count * 8 {
                for column in 0..packed.width {
                    let word = packed.values[(row / 64) * packed.width + column];
                    assert_eq!(
                        dense.values[row * dense.width + column],
                        F::from_bool((word >> (row % 64)) & 1 == 1),
                        "row={row} column={column}"
                    );
                }
            }
            if count < 8 {
                assert!(packed.values.iter().all(|word| word >> (count * 8) == 0));
            }
        }
    }
}

#[test]
fn compact_random_generators_match_the_wide_input_sequence() {
    let compact = Blake3BinaryCompactAir::default();
    let wide = Blake3BinaryAir::default();
    for count in [1, 8, 16] {
        let dense = compact.generate_random_trace_rows::<F>(count, 0);
        let packed = compact.generate_random_trace_packed::<F>(count);
        let original = wide.generate_random_trace_rows::<F>(count, 0);
        for input in 0..count {
            let row = dense.row_slice(8 * input + 6).unwrap();
            let row: &Blake3BinaryCompactCols<F> = (*row).borrow();
            let original = original.row_slice(input).unwrap();
            let original: &Blake3BinaryCols<F> = (*original).borrow();
            assert_eq!(row.compression_output(), original.compression_output());
        }
        for row in 0..count * 8 {
            for column in 0..packed.width {
                let word = packed.values[(row / 64) * packed.width + column];
                assert_eq!(
                    dense.values[row * dense.width + column],
                    F::from_bool((word >> (row % 64)) & 1 == 1)
                );
            }
        }
    }
}

#[test]
fn compact_generators_reject_invalid_input_shapes_and_characteristics() {
    for inputs in [vec![], vec![Blake3CompressionInput::default(); 3]] {
        assert!(
            std::panic::catch_unwind(|| generate_binary_compact_trace_rows::<F>(inputs.clone(), 0))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| generate_binary_compact_trace_packed::<F>(inputs)).is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(|| generate_binary_compact_trace_rows::<BabyBear>(
            vec![Blake3CompressionInput::default()],
            0
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| generate_binary_compact_trace_packed::<BabyBear>(vec![
            Blake3CompressionInput::default()
        ]))
        .is_err()
    );
}

#[test]
fn compact_dense_generator_reserves_requested_dft_capacity() {
    let trace = generate_binary_compact_trace_rows::<F>(vec![Blake3CompressionInput::default()], 2);
    assert_eq!(trace.height(), 8);
    assert!(trace.values.capacity() >= 4 * trace.values.len());
}
