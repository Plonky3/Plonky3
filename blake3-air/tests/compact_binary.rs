use core::array;
use core::borrow::{Borrow, BorrowMut};

use p3_air::{
    AirLayout, BaseAir, check_all_constraints, check_constraints, get_max_constraint_degree,
    get_symbolic_constraints,
};
use p3_binary_field::BinaryField128;
use p3_blake3_air::{
    BLAKE3_BINARY_COMPACT_OUTPUT_ROW, BLAKE3_BINARY_COMPACT_ROWS_PER_COMPRESSION,
    Blake3BinaryCompactAir, Blake3BinaryCompactCols, Blake3CompressionInput,
    NUM_BLAKE3_BINARY_COMPACT_COLS, generate_binary_compact_trace_rows,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const SCHEDULE: [[usize; 4]; 8] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

fn inputs(count: usize) -> Vec<Blake3CompressionInput> {
    let mut rng = SmallRng::seed_from_u64(314);
    (0..count)
        .map(|_| Blake3CompressionInput {
            chaining_value: rng.random(),
            block: rng.random(),
            counter: rng.random(),
            block_len: rng.random(),
            flags: rng.random(),
        })
        .collect()
}

fn reference_compress(input: &Blake3CompressionInput) -> [u32; 16] {
    let mut v = [0; 16];
    v[..8].copy_from_slice(&input.chaining_value);
    v[8..12].copy_from_slice(&IV[..4]);
    v[12..].copy_from_slice(&[
        input.counter as u32,
        (input.counter >> 32) as u32,
        input.block_len,
        input.flags,
    ]);
    let mut message = input.block;
    for _ in 0..7 {
        for (g, [a, b, c, d]) in SCHEDULE.into_iter().enumerate() {
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(message[2 * g]);
            v[d] = (v[d] ^ v[a]).rotate_right(16);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(12);
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(message[2 * g + 1]);
            v[d] = (v[d] ^ v[a]).rotate_right(8);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(7);
        }
        message = PERMUTATION.map(|i| message[i]);
    }
    array::from_fn(|i| {
        if i < 8 {
            v[i] ^ v[i + 8]
        } else {
            v[i] ^ input.chaining_value[i - 8]
        }
    })
}

// Independent algebraic bit-adder: this also constructs valid non-Boolean witnesses.
fn add(x: [F; 32], y: [F; 32]) -> ([F; 32], [F; 31]) {
    let mut carry = F::ZERO;
    let mut sum = [F::ZERO; 32];
    let mut carries = [F::ZERO; 31];
    for i in 0..32 {
        sum[i] = x[i] + y[i] + carry;
        carry = (x[i] + carry) * (y[i] + carry) + carry;
        if i < 31 {
            carries[i] = carry;
        }
    }
    (sum, carries)
}

fn xor_rotate(x: [F; 32], y: [F; 32], shift: usize) -> [F; 32] {
    array::from_fn(|i| x[(i + shift) % 32] + y[(i + shift) % 32])
}

fn fill_round(
    row: &mut Blake3BinaryCompactCols<F>,
    mut state: [[F; 32]; 16],
    message: [[F; 32]; 16],
    cv: [[F; 32]; 8],
) -> [[F; 32]; 16] {
    row.state = state;
    row.block = message;
    row.chaining_value = cv;
    for (g, [a, b, c, d]) in SCHEDULE.into_iter().enumerate() {
        let (ab, carry1) = add(state[a], state[b]);
        let (a1, _) = add(ab, message[2 * g]);
        let d1 = xor_rotate(state[d], a1, 16);
        let (c1, _) = add(state[c], d1);
        let b1 = xor_rotate(state[b], c1, 12);
        let (ab1, carry2) = add(a1, b1);
        let (a2, _) = add(ab1, message[2 * g + 1]);
        let d2 = xor_rotate(d1, a2, 8);
        let (c2, _) = add(c1, d2);
        let b2 = xor_rotate(b1, c2, 7);
        row.round[g].add1_carries = carry1;
        row.round[g].d1 = d1;
        row.round[g].b1 = b1;
        row.round[g].add2_carries = carry2;
        row.round[g].d2 = d2;
        row.round[g].b2 = b2;
        state[a] = a2;
        state[b] = b2;
        state[c] = c2;
        state[d] = d2;
    }
    state
}

fn rewrite_suffix(
    trace: &mut RowMajorMatrix<F>,
    start: usize,
    mut state: [[F; 32]; 16],
    mut message: [[F; 32]; 16],
    cv: [[F; 32]; 8],
) {
    let end = start / 8 * 8 + 7;
    for index in start..end {
        state = fill_round(trace.row_mut(index).borrow_mut(), state, message, cv);
        message = PERMUTATION.map(|i| message[i]);
    }
}

#[test]
fn compact_metadata_matches_actual_constraints_and_degree() {
    assert_eq!(NUM_BLAKE3_BINARY_COMPACT_COLS, 2_800);
    assert_eq!(BLAKE3_BINARY_COMPACT_ROWS_PER_COMPRESSION, 8);
    assert_eq!(BLAKE3_BINARY_COMPACT_OUTPUT_ROW, 6);
    for (air, boolean, count) in [
        (Blake3BinaryCompactAir::default(), false, 4_464),
        (
            Blake3BinaryCompactAir::assuming_boolean_trace(),
            true,
            3_184,
        ),
    ] {
        let layout = AirLayout::from_air::<F>(&air);
        assert_eq!(<Blake3BinaryCompactAir as BaseAir<F>>::width(&air), 2_800);
        assert_eq!(get_symbolic_constraints::<F, _>(&air, layout).len(), count);
        assert_eq!(
            <Blake3BinaryCompactAir as BaseAir<F>>::num_constraints(&air),
            Some(count)
        );
        assert_eq!(get_max_constraint_degree::<F, _>(&air, layout, 16), 2);
        assert_eq!(
            <Blake3BinaryCompactAir as BaseAir<F>>::max_constraint_degree(&air),
            Some(2)
        );
        assert_eq!(
            <Blake3BinaryCompactAir as BaseAir<F>>::assumes_boolean_trace(&air),
            boolean
        );
        assert_eq!(
            <Blake3BinaryCompactAir as BaseAir<F>>::main_next_row_columns(&air),
            (0..2_800).collect::<Vec<_>>()
        );
    }
}

#[test]
fn compact_full_outputs_match_independent_compression() {
    let mut data = inputs(16);
    data[0] = Blake3CompressionInput {
        chaining_value: [u32::MAX; 8],
        block: [u32::MAX; 16],
        counter: u64::MAX,
        block_len: u32::MAX,
        flags: u32::MAX,
    };
    let trace = generate_binary_compact_trace_rows::<F>(data.clone(), 0);
    assert_eq!(trace.height(), 128);
    for air in [
        Blake3BinaryCompactAir::default(),
        Blake3BinaryCompactAir::assuming_boolean_trace(),
    ] {
        check_constraints(&air, &trace, &[]);
    }
    for (index, input) in data.iter().enumerate() {
        let row = trace.row_slice(8 * index + 6).unwrap();
        let row: &Blake3BinaryCompactCols<F> = (*row).borrow();
        assert_eq!(row.compression_output(), reference_compress(input));
    }
}

#[test]
fn compact_single_block_hashes_match_blake3() {
    for len in [0, 1, 17, 32, 63, 64] {
        let message: Vec<u8> = (0..len).map(|i| (i * 31) as u8).collect();
        let mut padded = [0u8; 64];
        padded[..len].copy_from_slice(&message);
        let input = Blake3CompressionInput {
            chaining_value: IV,
            block: array::from_fn(|i| {
                u32::from_le_bytes(padded[4 * i..4 * i + 4].try_into().unwrap())
            }),
            counter: 0,
            block_len: len as u32,
            flags: 1 | 2 | 8,
        };
        let trace = generate_binary_compact_trace_rows::<F>(vec![input], 0);
        check_constraints(&Blake3BinaryCompactAir::default(), &trace, &[]);
        let row = trace.row_slice(6).unwrap();
        let cols: &Blake3BinaryCompactCols<F> = (*row).borrow();
        let output = cols.compression_output();
        let hash: [u8; 32] = array::from_fn(|i| output[i / 4].to_le_bytes()[i % 4]);
        assert_eq!(&hash, blake3::hash(&message).as_bytes());
        assert_eq!(output, reference_compress(&input));
    }
}

#[test]
fn compact_initialization_is_checked_at_each_compression() {
    for start in [0, 8] {
        let mut trace = generate_binary_compact_trace_rows::<F>(inputs(2), 0);
        let (mut state, message, cv) = {
            let row: &Blake3BinaryCompactCols<F> = (*trace.row_mut(start)).borrow();
            (row.state, row.block, row.chaining_value)
        };
        state[8][0] += F::ONE;
        rewrite_suffix(&mut trace, start, state, message, cv);
        let failures =
            check_all_constraints(&Blake3BinaryCompactAir::default(), &trace, &[], None).failures;
        assert!(!failures.is_empty());
        assert!(failures.iter().all(|failure| failure.row == start));
    }
}

#[test]
fn compact_transitions_bind_state_message_and_original_cv() {
    for (start, target) in [(3, 0), (6, 0), (3, 1), (6, 2)] {
        let mut trace = generate_binary_compact_trace_rows::<F>(inputs(2), 0);
        let (mut state, mut message, mut cv) = {
            let row: &Blake3BinaryCompactCols<F> = (*trace.row_mut(start)).borrow();
            (row.state, row.block, row.chaining_value)
        };
        match target {
            0 => state[0][0] += F::ONE,
            1 => message.swap(0, 1),
            _ => cv[0][0] += F::ONE,
        }
        rewrite_suffix(&mut trace, start, state, message, cv);
        let failures =
            check_all_constraints(&Blake3BinaryCompactAir::default(), &trace, &[], None).failures;
        assert!(!failures.is_empty(), "start={start} target={target}");
        assert!(failures.iter().all(|failure| failure.row == start - 1));
    }
}

#[test]
fn compact_padding_has_valid_round_constraints_and_no_boundary_links() {
    let mut trace = generate_binary_compact_trace_rows::<F>(inputs(2), 0);
    let (state, message, cv) = {
        let row: &Blake3BinaryCompactCols<F> = (*trace.row_mut(3)).borrow();
        (row.state, row.block, row.chaining_value)
    };
    for row in [7, 15] {
        let _ = fill_round(trace.row_mut(row).borrow_mut(), state, message, cv);
    }
    check_constraints(&Blake3BinaryCompactAir::default(), &trace, &[]);
    let row: &mut Blake3BinaryCompactCols<F> = trace.row_mut(15).borrow_mut();
    row.round[0].add1_carries[0] += F::ONE;
    let failures =
        check_all_constraints(&Blake3BinaryCompactAir::default(), &trace, &[], None).failures;
    assert!(!failures.is_empty());
    assert!(failures.iter().all(|failure| failure.row == 15));
}

#[test]
fn compact_default_requires_boolean_inputs_even_with_valid_rounds() {
    let mut trace = generate_binary_compact_trace_rows::<F>(inputs(1), 0);
    let (mut state, message, mut cv) = {
        let row: &Blake3BinaryCompactCols<F> = (*trace.row_mut(0)).borrow();
        (row.state, row.block, row.chaining_value)
    };
    cv[0][0] = F::GENERATOR;
    state[0][0] = F::GENERATOR;
    rewrite_suffix(&mut trace, 0, state, message, cv);
    assert!(
        check_all_constraints(
            &Blake3BinaryCompactAir::assuming_boolean_trace(),
            &trace,
            &[],
            None
        )
        .is_ok()
    );
    let failures =
        check_all_constraints(&Blake3BinaryCompactAir::default(), &trace, &[], None).failures;
    assert!(!failures.is_empty());
    assert!(failures.iter().all(|failure| failure.constraint < 1_280));
}

#[test]
fn compact_periodic_evaluations_match_the_repeated_tables() {
    let air = Blake3BinaryCompactAir::default();
    let columns = <Blake3BinaryCompactAir as BaseAir<F>>::periodic_columns(&air);
    let mut rng = SmallRng::seed_from_u64(918);
    for arity in [3, 4, 8] {
        let point: Vec<F> = (0..arity).map(|_| rng.random()).collect();
        let actual =
            <Blake3BinaryCompactAir as BaseAir<F>>::periodic_evaluations(&air, &point).unwrap();
        let expected: Vec<F> = columns
            .iter()
            .map(|column| {
                (0..1 << arity)
                    .map(|row| {
                        let weight = point.iter().enumerate().fold(
                            F::ONE,
                            |weight, (index, &coordinate)| {
                                weight
                                    * if row & (1 << (arity - index - 1)) == 0 {
                                        F::ONE - coordinate
                                    } else {
                                        coordinate
                                    }
                            },
                        );
                        column[row % column.len()] * weight
                    })
                    .sum()
            })
            .collect();
        assert_eq!(actual, expected);
    }
    for arity in 0..3 {
        assert!(
            <Blake3BinaryCompactAir as BaseAir<F>>::periodic_evaluations(
                &air,
                &vec![F::ZERO; arity]
            )
            .is_none()
        );
    }
}

#[test]
fn compact_trace_requires_complete_eight_row_groups() {
    let mut trace = generate_binary_compact_trace_rows::<F>(inputs(1), 0);
    trace.values.truncate(4 * trace.width);
    assert!(
        std::panic::catch_unwind(|| check_constraints(
            &Blake3BinaryCompactAir::default(),
            &trace,
            &[]
        ))
        .is_err()
    );
}
