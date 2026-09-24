use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::{Borrow, BorrowMut};

use blake2::digest::consts::U32;
use blake2::{Blake2s, Digest};
use hex_literal::hex;
use p3_air::{
    AirLayout, BaseAir, BaseEntry, BaseLeaf, ConstraintFailure, SymbolicExpr,
    check_all_constraints, check_constraints, get_max_constraint_degree, get_symbolic_constraints,
};
use p3_binary_field::BinaryField128;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::air::NUM_INPUT_BITS;
use super::{
    Blake2sBinaryAir, Blake2sBinaryCols, Blake2sCompressionInput, NUM_BLAKE2S_BINARY_COLS,
    generate_binary_trace_packed, generate_binary_trace_rows,
};
use crate::constants::{IV, SIGMA};

type F = BinaryField128;

/// Bytes hashed by one compression.
const BLOCK_BYTES: usize = 64;

/// The parameter block of an unkeyed BLAKE2s-256, XORed into the first word of the state.
///
/// It reads as `digest length | key length | fanout | depth`, so 32-byte digests, no key,
/// and the sequential tree shape.
const PARAM_BLOCK_0: u32 = 0x0101_0020;

/// BLAKE2s-256 of "abc", from RFC 7693 appendix B.
const ABC_DIGEST: [u8; 32] =
    hex!("508C5E8C327C14E2E1A72BA34EEB452F37458B209ED63A294D999B4C86675982");

/// The BLAKE2s G function on the state words at `a`, `b`, `c`, `d`.
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

/// The BLAKE2s compression function on plain words, written from RFC 7693 section 3.2.
fn reference_compress(input: &Blake2sCompressionInput) -> [u32; 8] {
    let cv = input.chaining_value;
    let last_block = if input.last_block { u32::MAX } else { 0 };
    let last_node = if input.last_node { u32::MAX } else { 0 };
    let mut v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ (input.counter as u32),
        IV[5] ^ ((input.counter >> 32) as u32),
        IV[6] ^ last_block,
        IV[7] ^ last_node,
    ];
    let m = input.block;
    for schedule in SIGMA {
        reference_g(&mut v, [0, 4, 8, 12], m[schedule[0]], m[schedule[1]]);
        reference_g(&mut v, [1, 5, 9, 13], m[schedule[2]], m[schedule[3]]);
        reference_g(&mut v, [2, 6, 10, 14], m[schedule[4]], m[schedule[5]]);
        reference_g(&mut v, [3, 7, 11, 15], m[schedule[6]], m[schedule[7]]);
        reference_g(&mut v, [0, 5, 10, 15], m[schedule[8]], m[schedule[9]]);
        reference_g(&mut v, [1, 6, 11, 12], m[schedule[10]], m[schedule[11]]);
        reference_g(&mut v, [2, 7, 8, 13], m[schedule[12]], m[schedule[13]]);
        reference_g(&mut v, [3, 4, 9, 14], m[schedule[14]], m[schedule[15]]);
    }
    array::from_fn(|i| cv[i] ^ v[i] ^ v[i + 8])
}

/// The message block starting at `offset`, zero padded, as sixteen little-endian words.
fn message_block(message: &[u8], offset: usize) -> [u32; 16] {
    let mut bytes = [0u8; BLOCK_BYTES];
    let end = (offset + BLOCK_BYTES).min(message.len());
    bytes[..end - offset].copy_from_slice(&message[offset..end]);
    array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}

/// Hash `message` by proving every compression of it, and return the digest.
///
/// A hash is a chain: each compression reads the previous one's output as its chaining value,
/// counts the bytes consumed so far, and only the last one sets the final-block flag. So each
/// block is proved on its own trace, since the next block's input does not exist until this
/// one's trace has been generated. An empty message still compresses one all-zero block.
fn proved_hash(message: &[u8]) -> [u8; 32] {
    let mut chaining_value = IV;
    chaining_value[0] ^= PARAM_BLOCK_0;

    let num_blocks = message.len().div_ceil(BLOCK_BYTES).max(1);
    for block in 0..num_blocks {
        let offset = block * BLOCK_BYTES;
        let last = block + 1 == num_blocks;
        let input = Blake2sCompressionInput {
            chaining_value,
            block: message_block(message, offset),
            counter: if last {
                message.len() as u64
            } else {
                (offset + BLOCK_BYTES) as u64
            },
            last_block: last,
            last_node: false,
        };
        chaining_value = trace_outputs(&[input])[0];
    }

    array::from_fn(|i| chaining_value[i / 4].to_le_bytes()[i % 4])
}

/// Generate a trace for `inputs`, check every constraint, and return the output of every row.
fn trace_outputs(inputs: &[Blake2sCompressionInput]) -> Vec<[u32; 8]> {
    let air = Blake2sBinaryAir::default();
    let padded = inputs.len().next_power_of_two();
    let mut padded_inputs = inputs.to_vec();
    padded_inputs.resize(padded, Blake2sCompressionInput::default());

    let trace = generate_binary_trace_rows::<F>(padded_inputs, 0);
    check_constraints(&air, &trace, &[]);
    (0..inputs.len())
        .map(|r| {
            let row = trace.row_slice(r).unwrap();
            let row: &Blake2sBinaryCols<F> = (*row).borrow();
            row.compression_output()
        })
        .collect()
}

/// Constraint failures of a random four-row trace after `mutate` edits row `row_index`.
///
/// Every failure must be on the edited row, since the AIR reads no next row.
fn failures_after_edit(
    row_index: usize,
    mutate: impl FnOnce(&mut Blake2sBinaryCols<F>),
) -> Vec<ConstraintFailure> {
    let air = Blake2sBinaryAir::default();
    let mut trace = air.generate_random_trace_rows::<F>(4, 0);
    mutate(trace.row_mut(row_index).borrow_mut());
    let failures = check_all_constraints(&air, &trace, &[], None).failures;
    assert!(failures.iter().all(|failure| failure.row == row_index));
    failures
}

#[test]
fn width_and_constraint_hints_match_symbolic_evaluation() {
    let air = Blake2sBinaryAir::default();
    assert_eq!(NUM_BLAKE2S_BINARY_COLS, 16_096);
    assert_eq!(<Blake2sBinaryAir as BaseAir<F>>::width(&air), 16_096);
    assert!(<Blake2sBinaryAir as BaseAir<F>>::main_next_row_columns(&air).is_empty());

    let layout = AirLayout {
        main_width: NUM_BLAKE2S_BINARY_COLS,
        ..Default::default()
    };
    for (air, assumes_boolean_trace, num_constraints) in [
        (Blake2sBinaryAir::default(), false, 16_096),
        (Blake2sBinaryAir::assuming_boolean_trace(), true, 15_200),
    ] {
        let constraints = get_symbolic_constraints::<F, _>(&air, layout);
        assert_eq!(constraints.len(), num_constraints);
        assert_eq!(
            <Blake2sBinaryAir as BaseAir<F>>::num_constraints(&air),
            Some(constraints.len())
        );
        assert_eq!(
            <Blake2sBinaryAir as BaseAir<F>>::assumes_boolean_trace(&air),
            assumes_boolean_trace
        );

        let degree = get_max_constraint_degree::<F, _>(&air, layout, 1 << 4);
        assert_eq!(degree, 2);
        assert_eq!(
            <Blake2sBinaryAir as BaseAir<F>>::max_constraint_degree(&air),
            Some(degree)
        );
    }
}

#[test]
fn the_rfc_7693_digest_of_abc_is_proved() {
    assert_eq!(proved_hash(b"abc"), ABC_DIGEST);
}

#[test]
fn proved_hashes_match_the_blake2_crate() {
    let mut rng = SmallRng::seed_from_u64(2);
    // One partial block, a message that ends exactly on a block, and messages that need two
    // and three compressions, so the counter and the chaining both move.
    for length in [0, 1, 55, 63, 64, 65, 100, 128, 129, 200] {
        let message: Vec<u8> = (0..length).map(|_| rng.random()).collect();
        let expected: [u8; 32] = Blake2s::<U32>::digest(&message).into();
        assert_eq!(proved_hash(&message), expected, "message of {length} bytes");
    }
}

#[test]
fn random_inputs_match_the_reference_compression() {
    let mut rng = SmallRng::seed_from_u64(3);
    let inputs: Vec<_> = (0..16)
        .map(|_| Blake2sCompressionInput {
            chaining_value: rng.random(),
            block: rng.random(),
            counter: rng.random(),
            last_block: rng.random(),
            last_node: rng.random(),
        })
        .collect();

    for (input, output) in inputs.iter().zip(trace_outputs(&inputs)) {
        assert_eq!(output, reference_compress(input));
    }
}

#[test]
fn random_traces_satisfy_constraints() {
    let air = Blake2sBinaryAir::default();
    for height in [1, 2, 4] {
        let trace = air.generate_random_trace_rows::<F>(height, 0);
        assert_eq!(trace.height(), height);
        check_constraints(&air, &trace, &[]);
    }
}

#[test]
fn packed_trace_matches_dense_trace_at_word_boundaries() {
    let mut rng = SmallRng::seed_from_u64(17);
    // All-ones inputs carry out of every bit of the first additions, so the full carry chains
    // are exercised whatever the random draws.
    let all_ones = Blake2sCompressionInput {
        chaining_value: [u32::MAX; 8],
        block: [u32::MAX; 16],
        counter: u64::MAX,
        last_block: true,
        last_node: true,
    };
    for height in [1usize, 2, 32, 64, 128] {
        let random: Vec<_> = (0..height)
            .map(|_| Blake2sCompressionInput {
                chaining_value: rng.random(),
                block: rng.random(),
                counter: rng.random(),
                last_block: rng.random(),
                last_node: rng.random(),
            })
            .collect();
        for inputs in [random, vec![all_ones; height]] {
            let dense = generate_binary_trace_rows::<F>(inputs.clone(), 0);
            let packed = generate_binary_trace_packed::<F>(inputs);
            assert_eq!(packed.width, NUM_BLAKE2S_BINARY_COLS);
            assert_eq!(packed.height(), height.div_ceil(64));
            for row in 0..height {
                for column in 0..NUM_BLAKE2S_BINARY_COLS {
                    let expected = dense.values[row * NUM_BLAKE2S_BINARY_COLS + column];
                    let word = packed.values[(row / 64) * packed.width + column];
                    assert_eq!(expected, F::from_bool((word >> (row % 64)) & 1 == 1));
                }
            }
            if !height.is_multiple_of(64) {
                for word in &packed.values[packed.width * (packed.height() - 1)..] {
                    assert_eq!(*word >> (height % 64), 0);
                }
            }
        }
    }
}

#[test]
fn packed_random_trace_uses_the_dense_generator_sequence() {
    let air = Blake2sBinaryAir::default();
    for height in [1usize, 64, 128] {
        let dense = air.generate_random_trace_rows::<F>(height, 0);
        let packed = air.generate_random_trace_packed::<F>(height);
        for row in 0..height {
            for column in 0..NUM_BLAKE2S_BINARY_COLS {
                let word = packed.values[(row / 64) * packed.width + column];
                assert_eq!(
                    dense.values[row * NUM_BLAKE2S_BINARY_COLS + column],
                    F::from_bool((word >> (row % 64)) & 1 == 1)
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "at least one input")]
fn packed_generator_rejects_empty_input() {
    let _ = generate_binary_trace_packed::<F>(Vec::new());
}

#[test]
#[should_panic(expected = "power of two")]
fn packed_generator_rejects_non_power_of_two_input() {
    let _ = generate_binary_trace_packed::<F>(vec![Blake2sCompressionInput::default(); 3]);
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
        let failures = failures_after_edit(2, |row| row.rounds[9][g].b2[3 * g] += F::ONE);
        assert!(!failures.is_empty(), "b2 of step {g}");
        let failures = failures_after_edit(3, |row| row.rounds[9][g].d2[31 - g] += F::ONE);
        assert!(!failures.is_empty(), "d2 of step {g}");
    }
}

#[test]
fn rejects_a_flipped_counter_or_flag_bit() {
    // The counter and the flags reach the state only through the initial `d` words, so a flip
    // in one of them has to break the first round's additions to be caught at all.
    let failures = failures_after_edit(1, |row| row.counter_low[9] += F::ONE);
    assert!(!failures.is_empty(), "counter low");
    let failures = failures_after_edit(2, |row| row.counter_high[0] += F::ONE);
    assert!(!failures.is_empty(), "counter high");
    let failures = failures_after_edit(3, |row| row.last_block[31] += F::ONE);
    assert!(!failures.is_empty(), "last block flag");
    let failures = failures_after_edit(0, |row| row.last_node[17] += F::ONE);
    assert!(!failures.is_empty(), "last node flag");
}

#[test]
fn rejects_non_boolean_input() {
    assert!(F::GENERATOR != F::ZERO && F::GENERATOR != F::ONE);
    // Inputs are checked first in the order chaining value, block, counter low, counter high,
    // last block, last node, so bit `b` of input word `w` is constraint `32 * w + b`.
    for word in 0..28 {
        let bit = (7 * word) % 32;
        let failures = failures_after_edit(1, |row| {
            input_word(row, word)[bit] = F::GENERATOR;
        });
        assert!(
            failures
                .iter()
                .any(|failure| failure.constraint == 32 * word + bit),
            "bit {bit} of input word {word}"
        );
    }
}

#[test]
fn rejects_non_boolean_witness_at_the_top_bit() {
    // The carry into bit 31 of a 3-operand addition.
    let failures = failures_after_edit(2, |row| row.rounds[2][6].add1_carries[30] = F::GENERATOR);
    assert!(!failures.is_empty());
    // Bit 23 of `d2` feeds bit 31 of `a2 = d1 ^ (d2 <<< 8)`.
    let failures = failures_after_edit(3, |row| row.rounds[4][1].d2[23] = F::GENERATOR);
    assert!(!failures.is_empty());
}

#[test]
fn rejects_message_bit_flip_without_regeneration() {
    let failures = failures_after_edit(2, |row| row.block[5][9] += F::ONE);
    assert!(!failures.is_empty());
}

/// The input word at `index`, in the order the booleanity constraints check them.
fn input_word(row: &mut Blake2sBinaryCols<F>, index: usize) -> &mut [F; 32] {
    match index {
        0..8 => &mut row.chaining_value[index],
        8..24 => &mut row.block[index - 8],
        24 => &mut row.counter_low,
        25 => &mut row.counter_high,
        26 => &mut row.last_block,
        _ => &mut row.last_node,
    }
}

/// Every current-row main column the expression reads.
fn columns_of(expr: &SymbolicExpr<BaseLeaf<F>>, out: &mut BTreeSet<usize>) {
    match expr {
        SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => {
            if v.entry == (BaseEntry::Main { offset: 0 }) {
                out.insert(v.index);
            }
        }
        SymbolicExpr::Leaf(_) => {}
        SymbolicExpr::Add { x, y, .. }
        | SymbolicExpr::Sub { x, y, .. }
        | SymbolicExpr::Mul { x, y, .. } => {
            columns_of(x, out);
            columns_of(y, out);
        }
        SymbolicExpr::Neg { x, .. } => columns_of(x, out),
    }
}

/// Evaluates `expr` on the cells of one row.
fn eval_on_row(expr: &SymbolicExpr<BaseLeaf<F>>, row: &[F]) -> F {
    match expr {
        SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => {
            assert_eq!(v.entry, BaseEntry::Main { offset: 0 });
            row[v.index]
        }
        SymbolicExpr::Leaf(BaseLeaf::Constant(c)) => *c,
        SymbolicExpr::Leaf(_) => unreachable!("the AIR uses no selector"),
        SymbolicExpr::Add { x, y, .. } => eval_on_row(x, row) + eval_on_row(y, row),
        SymbolicExpr::Sub { x, y, .. } => eval_on_row(x, row) - eval_on_row(y, row),
        SymbolicExpr::Mul { x, y, .. } => eval_on_row(x, row) * eval_on_row(y, row),
        SymbolicExpr::Neg { x, .. } => -eval_on_row(x, row),
    }
}

/// Sets every non-input cell of `row` to the value the constraints give it from the inputs.
///
/// Each constraint after input booleanity is one new column plus an expression in columns
/// already set, so in characteristic 2 that column equals the expression. No cell has to be a
/// bit for this.
fn derive_from_inputs(row: &mut [F]) {
    let layout = AirLayout {
        main_width: NUM_BLAKE2S_BINARY_COLS,
        ..Default::default()
    };
    let constraints = get_symbolic_constraints::<F, _>(&Blake2sBinaryAir::default(), layout);

    let mut fixed: BTreeSet<usize> = BTreeSet::new();
    for (i, constraint) in constraints.iter().enumerate() {
        let mut columns = BTreeSet::new();
        columns_of(constraint, &mut columns);
        if i < NUM_INPUT_BITS {
            fixed.extend(columns);
            continue;
        }

        let new: Vec<usize> = columns.difference(&fixed).copied().collect();
        let [column] = new[..] else {
            panic!("constraint {i} introduces columns {new:?}");
        };
        row[column] = F::ZERO;
        row[column] = eval_on_row(constraint, row);
        fixed.insert(column);
    }
}

#[test]
fn only_booleanity_rejects_a_non_boolean_input() {
    // A non-bit input cell, with every other cell derived from the inputs, fails only its own
    // booleanity constraint. The AIR without booleanity accepts the row, so nothing else keeps
    // an input cell in `{0, 1}`.
    //
    // Bit `b` of input word `w` is constraint `32 * w + b`, as in `rejects_non_boolean_input`.
    assert!(F::GENERATOR != F::ZERO && F::GENERATOR != F::ONE);
    let air = Blake2sBinaryAir::default();
    for (word, bit) in [(0, 0), (13, 31), (27, 17)] {
        let mut trace = air.generate_random_trace_rows::<F>(1, 0);
        let row = trace.row_mut(0);
        let cols: &mut Blake2sBinaryCols<F> = row.borrow_mut();
        input_word(cols, word)[bit] = F::GENERATOR;
        derive_from_inputs(row);

        // The non-bit reached the derived cells.
        let non_bits = row.iter().filter(|&&c| c != F::ZERO && c != F::ONE).count();
        assert!(non_bits > 1, "bit {bit} of input word {word}");

        let failures: Vec<usize> = check_all_constraints(&air, &trace, &[], None)
            .failures
            .iter()
            .map(|failure| failure.constraint)
            .collect();
        assert_eq!(
            failures,
            [32 * word + bit],
            "bit {bit} of input word {word}"
        );

        let unconstrained = Blake2sBinaryAir::assuming_boolean_trace();
        assert!(
            check_all_constraints(&unconstrained, &trace, &[], None).is_ok(),
            "bit {bit} of input word {word}"
        );
    }
}
