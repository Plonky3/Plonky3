use alloc::vec::Vec;
use core::array;
use core::borrow::{Borrow, BorrowMut};

use p3_air::{
    AirLayout, BaseAir, ConstraintFailure, check_all_constraints, check_constraints,
    get_max_constraint_degree, get_symbolic_constraints,
};
use p3_baby_bear::BabyBear;
use p3_binary_field::BinaryField128;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::{
    Blake3BinaryAir, Blake3BinaryCols, Blake3CompressionInput, G_SCHEDULE, NUM_BLAKE3_BINARY_COLS,
    generate_binary_trace_rows, iv_word,
};
use crate::constants::permute;

type F = BinaryField128;

/// Flags of a block that is the first, last and root chunk of a hash.
const SINGLE_BLOCK_FLAGS: u32 = 1 | 2 | 8;

/// Blake-3 hash of the empty input.
const EMPTY_INPUT_HASH: [u8; 32] = [
    0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6, 0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49,
    0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7, 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62,
];

/// The Blake-3 G function on the state words at `a`, `b`, `c`, `d`.
fn reference_g(v: &mut [u32; 16], [a, b, c, d]: [usize; 4], mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// The Blake-3 compression function on plain words.
fn reference_compress(input: &Blake3CompressionInput) -> [u32; 16] {
    let cv = input.chaining_value;
    let mut v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        iv_word(0),
        iv_word(1),
        iv_word(2),
        iv_word(3),
        input.counter as u32,
        (input.counter >> 32) as u32,
        input.block_len,
        input.flags,
    ];
    let mut m = input.block;
    for round in 0..7 {
        reference_g(&mut v, [0, 4, 8, 12], m[0], m[1]);
        reference_g(&mut v, [1, 5, 9, 13], m[2], m[3]);
        reference_g(&mut v, [2, 6, 10, 14], m[4], m[5]);
        reference_g(&mut v, [3, 7, 11, 15], m[6], m[7]);
        reference_g(&mut v, [0, 5, 10, 15], m[8], m[9]);
        reference_g(&mut v, [1, 6, 11, 12], m[10], m[11]);
        reference_g(&mut v, [2, 7, 8, 13], m[12], m[13]);
        reference_g(&mut v, [3, 4, 9, 14], m[14], m[15]);
        if round < 6 {
            permute(&mut m);
        }
    }
    array::from_fn(|i| {
        if i < 8 {
            v[i] ^ v[i + 8]
        } else {
            v[i] ^ cv[i - 8]
        }
    })
}

/// Read a word stored as 32 boolean cells.
fn read_word(bits: &[F; 32]) -> u32 {
    bits.iter().enumerate().fold(0, |word, (i, &bit)| {
        assert!(bit == F::ZERO || bit == F::ONE, "cell is not a bit");
        word | (u32::from(bit == F::ONE) << i)
    })
}

/// Recompute the compression output of a trace row from its witness columns.
///
/// The final state words are `a = d1 ^ (d2 <<< 8)`, `b = b2`, `c = b1 ^ (b2 <<< 7)` and
/// `d = d2` of the last G step that wrote them, and the output is
/// `v[i] ^ v[i + 8]` followed by `v[i + 8] ^ cv[i]`.
fn compression_output(row: &Blake3BinaryCols<F>) -> [u32; 16] {
    let mut v = [0u32; 16];
    for (cols, [ia, ib, ic, id]) in row.rounds[6].iter().zip(G_SCHEDULE) {
        let (d1, b1, d2, b2) = (
            read_word(&cols.d1),
            read_word(&cols.b1),
            read_word(&cols.d2),
            read_word(&cols.b2),
        );
        v[ia] = d1 ^ d2.rotate_left(8);
        v[4 + ib] = b2;
        v[8 + ic] = b1 ^ b2.rotate_left(7);
        v[12 + id] = d2;
    }
    let cv = row.chaining_value.map(|word| read_word(&word));
    array::from_fn(|i| {
        if i < 8 {
            v[i] ^ v[i + 8]
        } else {
            v[i] ^ cv[i - 8]
        }
    })
}

/// A single-block hash input for a message of at most 64 bytes.
fn single_block_input(message: &[u8]) -> Blake3CompressionInput {
    let mut bytes = [0u8; 64];
    bytes[..message.len()].copy_from_slice(message);
    Blake3CompressionInput {
        chaining_value: array::from_fn(iv_word),
        block: array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())),
        counter: 0,
        block_len: message.len() as u32,
        flags: SINGLE_BLOCK_FLAGS,
    }
}

/// Little-endian bytes of the first eight output words.
fn hash_bytes(output: &[u32; 16]) -> [u8; 32] {
    array::from_fn(|i| output[i / 4].to_le_bytes()[i % 4])
}

/// Generate a trace for `inputs` and return the output of every row.
fn trace_outputs(inputs: &[Blake3CompressionInput]) -> Vec<[u32; 16]> {
    let air = Blake3BinaryAir {};
    let trace = generate_binary_trace_rows::<F>(inputs.to_vec(), 0);
    check_constraints(&air, &trace, &[]);
    (0..trace.height())
        .map(|r| compression_output((*trace.row_slice(r).unwrap()).borrow()))
        .collect()
}

/// Constraint failures of a random four-row trace after `mutate` edits row `row_index`.
///
/// Every failure must be on the edited row, since the AIR reads no next row.
fn failures_after_edit(
    row_index: usize,
    mutate: impl FnOnce(&mut Blake3BinaryCols<F>),
) -> Vec<ConstraintFailure> {
    let air = Blake3BinaryAir {};
    let mut trace = air.generate_random_trace_rows::<F>(4, 0);
    mutate(trace.row_mut(row_index).borrow_mut());
    let failures = check_all_constraints(&air, &trace, &[], None).failures;
    assert!(failures.iter().all(|failure| failure.row == row_index));
    failures
}

#[test]
fn width_and_constraint_hints_match_symbolic_evaluation() {
    let air = Blake3BinaryAir {};
    assert_eq!(NUM_BLAKE3_BINARY_COLS, 11_536);
    assert_eq!(<Blake3BinaryAir as BaseAir<F>>::width(&air), 11_536);
    assert!(<Blake3BinaryAir as BaseAir<F>>::main_next_row_columns(&air).is_empty());

    let layout = AirLayout {
        main_width: NUM_BLAKE3_BINARY_COLS,
        ..Default::default()
    };
    let constraints = get_symbolic_constraints::<F, _>(&air, layout);
    assert_eq!(constraints.len(), 11_536);
    assert_eq!(
        <Blake3BinaryAir as BaseAir<F>>::num_constraints(&air),
        Some(constraints.len())
    );

    let degree = get_max_constraint_degree::<F, _>(&air, layout, 1 << 4);
    assert_eq!(degree, 2);
    assert_eq!(
        <Blake3BinaryAir as BaseAir<F>>::max_constraint_degree(&air),
        Some(degree)
    );
}

#[test]
fn empty_input_known_answer() {
    let input = Blake3CompressionInput {
        chaining_value: array::from_fn(iv_word),
        block: [0; 16],
        counter: 0,
        block_len: 0,
        flags: SINGLE_BLOCK_FLAGS,
    };
    let outputs = trace_outputs(&[input]);
    assert_eq!(hash_bytes(&outputs[0]), EMPTY_INPUT_HASH);
    assert_eq!(outputs[0], reference_compress(&input));
}

#[test]
fn single_block_hashes_match_blake3() {
    let mut rng = SmallRng::seed_from_u64(2);
    let lengths = [0, 1, 4, 17, 32, 55, 63, 64];
    let messages: Vec<Vec<u8>> = lengths
        .iter()
        .map(|&len| (0..len).map(|_| rng.random()).collect())
        .collect();
    let inputs: Vec<_> = messages.iter().map(|m| single_block_input(m)).collect();

    for (message, output) in messages.iter().zip(trace_outputs(&inputs)) {
        assert_eq!(&hash_bytes(&output), blake3::hash(message).as_bytes());
    }
}

#[test]
fn random_inputs_match_reference_compression() {
    let mut rng = SmallRng::seed_from_u64(3);
    let inputs: Vec<_> = (0..16)
        .map(|_| Blake3CompressionInput {
            chaining_value: rng.random(),
            block: rng.random(),
            counter: rng.random(),
            block_len: rng.random(),
            flags: rng.random(),
        })
        .collect();

    for (input, output) in inputs.iter().zip(trace_outputs(&inputs)) {
        assert_eq!(output, reference_compress(input));
    }
}

#[test]
fn random_traces_satisfy_constraints() {
    let air = Blake3BinaryAir {};
    for height in [1, 2, 4] {
        let trace = air.generate_random_trace_rows::<F>(height, 0);
        assert_eq!(trace.height(), height);
        check_constraints(&air, &trace, &[]);
    }
}

#[test]
#[should_panic(expected = "characteristic 2")]
fn generator_rejects_odd_characteristic() {
    let air = Blake3BinaryAir {};
    air.generate_random_trace_rows::<BabyBear>(1, 0);
}

#[test]
fn rejects_flipped_carry_bit() {
    let failures = failures_after_edit(1, |row| row.rounds[3][5].add2_carries[17] += F::ONE);
    assert!(!failures.is_empty());
}

#[test]
fn rejects_flipped_d1_bit() {
    let failures = failures_after_edit(0, |row| row.rounds[0][2].d1[0] += F::ONE);
    assert!(!failures.is_empty());
}

#[test]
fn rejects_flipped_output_bits_in_last_round() {
    for g in 0..8 {
        let failures = failures_after_edit(2, |row| row.rounds[6][g].b2[3 * g] += F::ONE);
        assert!(!failures.is_empty(), "b2 of step {g}");
        let failures = failures_after_edit(3, |row| row.rounds[6][g].d2[31 - g] += F::ONE);
        assert!(!failures.is_empty(), "d2 of step {g}");
    }
}

#[test]
fn rejects_non_boolean_input() {
    assert!(F::GENERATOR != F::ZERO && F::GENERATOR != F::ONE);
    let failures = failures_after_edit(1, |row| row.chaining_value[1][3] = F::GENERATOR);
    // Inputs are checked first, so bit 3 of the second chaining value word is constraint 35.
    assert!(failures.iter().any(|failure| failure.constraint == 35));
}

#[test]
fn rejects_message_bit_flip_without_regeneration() {
    let failures = failures_after_edit(2, |row| row.block[5][9] += F::ONE);
    assert!(!failures.is_empty());
}
