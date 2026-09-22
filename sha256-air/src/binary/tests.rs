use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::array;
use core::borrow::{Borrow, BorrowMut};

use p3_air::{
    AirLayout, BaseAir, BaseEntry, BaseLeaf, ConstraintFailure, SymbolicExpr,
    check_all_constraints, check_constraints, get_max_constraint_degree, get_symbolic_constraints,
};
use p3_baby_bear::BabyBear;
use p3_binary_field::BinaryField128;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::air::{CONSTRAINTS_PER_ROUND, CONSTRAINTS_PER_SCHEDULE_WORD, NUM_INPUT_BITS};
use super::{
    NUM_SHA256_BINARY_COLS, Sha256BinaryAir, Sha256BinaryCols, generate_binary_trace_rows,
};
use crate::{INPUT_WORDS, SCHEDULE_EXTENSIONS, SHA256_IV};

type F = BinaryField128;

/// An in-place edit of one trace row.
type RowEdit = fn(&mut Sha256BinaryCols<F>);

/// The `sha2` crate's compression of one block from `h_in`.
fn reference_compress(input: &[u32; INPUT_WORDS]) -> [u32; 8] {
    let mut block = [0u8; 64];
    for (bytes, word) in block.as_chunks_mut::<4>().0.iter_mut().zip(&input[..16]) {
        *bytes = word.to_be_bytes();
    }
    let mut state: [u32; 8] = array::from_fn(|i| input[16 + i]);
    sha2::block_api::compress256(&mut state, core::slice::from_ref(&block));
    state
}

/// The single padded block of a message of at most 55 bytes, compressed from the IV.
fn single_block_input(message: &[u8]) -> [u32; INPUT_WORDS] {
    assert!(message.len() <= 55);
    let mut bytes = [0u8; 64];
    bytes[..message.len()].copy_from_slice(message);
    bytes[message.len()] = 0x80;
    bytes[56..].copy_from_slice(&(message.len() as u64 * 8).to_be_bytes());
    array::from_fn(|i| {
        if i < 16 {
            u32::from_be_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
        } else {
            SHA256_IV[i - 16]
        }
    })
}

/// Big-endian bytes of a chaining value, which is the digest of a single-block message.
fn digest_bytes(state: &[u32; 8]) -> [u8; 32] {
    array::from_fn(|i| state[i / 4].to_be_bytes()[i % 4])
}

/// Generate a trace for `inputs`, check its constraints, and return the output of every row.
fn trace_outputs(inputs: &[[u32; INPUT_WORDS]]) -> Vec<[u32; 8]> {
    let air = Sha256BinaryAir::default();
    let trace = generate_binary_trace_rows::<F>(inputs.to_vec(), 0);
    check_constraints(&air, &trace, &[]);
    (0..trace.height())
        .map(|r| {
            let row = trace.row_slice(r).unwrap();
            let row: &Sha256BinaryCols<F> = (*row).borrow();
            row.compression_output()
        })
        .collect()
}

/// Constraint failures of a random four-row trace after `mutate` edits row `row_index`.
///
/// Every failure must be on the edited row, since the AIR reads no next row.
fn failures_after_edit(
    row_index: usize,
    mutate: impl FnOnce(&mut Sha256BinaryCols<F>),
) -> Vec<ConstraintFailure> {
    let air = Sha256BinaryAir::default();
    let mut trace = air.generate_random_trace_rows::<F>(4, 0);
    mutate(trace.row_mut(row_index).borrow_mut());
    let failures = check_all_constraints(&air, &trace, &[], None).failures;
    assert!(failures.iter().all(|failure| failure.row == row_index));
    failures
}

#[test]
fn width_and_constraint_hints_match_symbolic_evaluation() {
    let air = Sha256BinaryAir::default();
    assert_eq!(NUM_SHA256_BINARY_COLS, 23_712);
    assert_eq!(<Sha256BinaryAir as BaseAir<F>>::width(&air), 23_712);
    assert!(<Sha256BinaryAir as BaseAir<F>>::main_next_row_columns(&air).is_empty());

    let layout = AirLayout {
        main_width: NUM_SHA256_BINARY_COLS,
        ..Default::default()
    };
    for (air, assumes_boolean_trace, num_constraints) in [
        (Sha256BinaryAir::default(), false, 23_712),
        (Sha256BinaryAir::assuming_boolean_trace(), true, 22_944),
    ] {
        let constraints = get_symbolic_constraints::<F, _>(&air, layout);
        assert_eq!(constraints.len(), num_constraints);
        assert_eq!(
            <Sha256BinaryAir as BaseAir<F>>::num_constraints(&air),
            Some(constraints.len())
        );
        assert_eq!(
            <Sha256BinaryAir as BaseAir<F>>::assumes_boolean_trace(&air),
            assumes_boolean_trace
        );

        let degree = get_max_constraint_degree::<F, _>(&air, layout, 1 << 4);
        assert_eq!(degree, 2);
        assert_eq!(
            <Sha256BinaryAir as BaseAir<F>>::max_constraint_degree(&air),
            Some(degree)
        );
    }
}

/// How a constraint depends on one column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    /// The column does not appear.
    Free,
    /// The constraint is the column plus an expression that does not contain it.
    Linear,
    /// Anything else, including a higher power of the column.
    Other,
}

/// Classifies how `expr` depends on main column `col`, over a field of characteristic 2.
fn form_in(expr: &SymbolicExpr<BaseLeaf<F>>, col: usize) -> Form {
    match expr {
        SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => {
            match (v.entry, v.index == col) {
                // Only current-row main columns can fix a cell of this row.
                (BaseEntry::Main { offset: 0 }, true) => Form::Linear,
                _ => Form::Free,
            }
        }
        SymbolicExpr::Leaf(_) => Form::Free,
        // In characteristic 2 subtraction is addition, and `x + x` cancels.
        SymbolicExpr::Add { x, y, .. } | SymbolicExpr::Sub { x, y, .. } => {
            match (form_in(x, col), form_in(y, col)) {
                (Form::Other, _) | (_, Form::Other) => Form::Other,
                (Form::Linear, Form::Linear) | (Form::Free, Form::Free) => Form::Free,
                _ => Form::Linear,
            }
        }
        SymbolicExpr::Neg { x, .. } => form_in(x, col),
        SymbolicExpr::Mul { x, y, .. } => match (form_in(x, col), form_in(y, col)) {
            (Form::Free, Form::Free) => Form::Free,
            // A linear factor survives only when multiplied by the constant one.
            (Form::Linear, Form::Free) if is_one(y) => Form::Linear,
            (Form::Free, Form::Linear) if is_one(x) => Form::Linear,
            _ => Form::Other,
        },
    }
}

/// Whether the expression is the constant one.
fn is_one(expr: &SymbolicExpr<BaseLeaf<F>>) -> bool {
    matches!(expr, SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if *c == F::ONE)
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
        SymbolicExpr::Add { x, y, .. } | SymbolicExpr::Sub { x, y, .. } => {
            columns_of(x, out);
            columns_of(y, out);
        }
        SymbolicExpr::Mul { x, y, .. } => {
            columns_of(x, out);
            columns_of(y, out);
        }
        SymbolicExpr::Neg { x, .. } => columns_of(x, out),
    }
}

#[test]
fn every_column_is_forced_to_a_bit_by_the_constraints() {
    // The AIR only asserts booleanity on the input columns and relies on the rest being forced
    // to bits by the constraints that define them. This walks all 23,712 constraints in order
    // and checks that claim for the whole row.
    //
    // After the input booleanity constraints, each remaining constraint must introduce exactly
    // one column no earlier constraint fixed, and must be linear in it. Then that column equals
    // an expression in columns already known to be bits, so it is a bit too, and the row is
    // determined by the inputs.
    let air = Sha256BinaryAir::default();
    let layout = AirLayout {
        main_width: NUM_SHA256_BINARY_COLS,
        ..Default::default()
    };
    let constraints = get_symbolic_constraints::<F, _>(&air, layout);

    let mut fixed: BTreeSet<usize> = BTreeSet::new();
    for (i, constraint) in constraints.iter().enumerate() {
        let mut columns = BTreeSet::new();
        columns_of(constraint, &mut columns);

        if i < NUM_INPUT_BITS {
            // An input booleanity constraint: one column, and not linear in it.
            assert_eq!(columns.len(), 1, "input constraint {i} reads {columns:?}");
            let column = *columns.first().unwrap();
            assert_eq!(form_in(constraint, column), Form::Other, "constraint {i}");
            fixed.insert(column);
            continue;
        }

        let new: Vec<usize> = columns.difference(&fixed).copied().collect();
        assert_eq!(new.len(), 1, "constraint {i} introduces columns {new:?}");
        assert_eq!(
            form_in(constraint, new[0]),
            Form::Linear,
            "constraint {i} is not linear in column {}",
            new[0]
        );
        fixed.insert(new[0]);
    }

    // Every column of the row ends up determined.
    assert_eq!(fixed.len(), NUM_SHA256_BINARY_COLS);
}

#[test]
fn fips_180_single_block_known_answers() {
    // FIPS 180-4 examples, plus the empty message.
    let vectors: [(&[u8], [u8; 32]); 2] = [
        (
            b"",
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
                0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
                0x78, 0x52, 0xb8, 0x55,
            ],
        ),
        (
            b"abc",
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ],
        ),
    ];
    let inputs: Vec<_> = vectors.iter().map(|(m, _)| single_block_input(m)).collect();
    for ((message, expected), output) in vectors.iter().zip(trace_outputs(&inputs)) {
        assert_eq!(digest_bytes(&output), *expected, "message {message:?}");
    }
}

#[test]
fn random_inputs_match_reference_compression() {
    let mut rng = SmallRng::seed_from_u64(3);
    let inputs: Vec<[u32; INPUT_WORDS]> = (0..16).map(|_| rng.random()).collect();
    for (input, output) in inputs.iter().zip(trace_outputs(&inputs)) {
        assert_eq!(output, reference_compress(input));
    }
}

#[test]
fn boundary_inputs_match_reference_compression() {
    // All-zero and all-ones inputs drive every addition to its extremes, and the IV with a
    // zero block is the standard starting point.
    let inputs = [
        [0u32; INPUT_WORDS],
        [u32::MAX; INPUT_WORDS],
        array::from_fn(|i| if i < 16 { 0 } else { SHA256_IV[i - 16] }),
        array::from_fn(|i| 1u32 << (i % 32)),
    ];
    for (input, output) in inputs.iter().zip(trace_outputs(&inputs)) {
        assert_eq!(output, reference_compress(input));
    }
}

#[test]
fn random_traces_satisfy_constraints() {
    let air = Sha256BinaryAir::default();
    for height in [1, 2, 4] {
        let trace = air.generate_random_trace_rows::<F>(height, 0);
        assert_eq!(trace.height(), height);
        check_constraints(&air, &trace, &[]);
    }
}

#[test]
#[should_panic(expected = "characteristic 2")]
fn generator_rejects_odd_characteristic() {
    let air = Sha256BinaryAir::default();
    air.generate_random_trace_rows::<BabyBear>(1, 0);
}

#[test]
fn rejects_flipped_round_witness_bits() {
    let flips: [(&str, RowEdit); 6] = [
        ("ch", |row| row.rounds[5].ch[9] += F::ONE),
        ("maj", |row| row.rounds[40].maj[31] += F::ONE),
        ("t1 carry", |row| row.rounds[17].t1_carries[2][30] += F::ONE),
        ("t1", |row| row.rounds[63].t1[0] += F::ONE),
        ("new a carry", |row| {
            row.rounds[31].new_a_carries[13] += F::ONE;
        }),
        ("schedule carry", |row| {
            row.schedule[47].carries[1][5] += F::ONE;
        }),
    ];
    for (name, flip) in flips {
        assert!(!failures_after_edit(1, flip).is_empty(), "{name}");
    }
}

#[test]
fn rejects_flipped_chain_schedule_and_output_bits() {
    for round in [0, 1, 32, 63] {
        let failures = failures_after_edit(2, |row| row.a_chain[round + 4][round % 32] += F::ONE);
        assert!(!failures.is_empty(), "a_chain after round {round}");
        let failures =
            failures_after_edit(3, |row| row.e_chain[round + 4][31 - round % 32] += F::ONE);
        assert!(!failures.is_empty(), "e_chain after round {round}");
    }
    for t in [16, 40, 63] {
        let failures = failures_after_edit(0, |row| row.w[t][t % 32] += F::ONE);
        assert!(!failures.is_empty(), "schedule word {t}");
    }
    for word in 0..8 {
        let failures = failures_after_edit(1, |row| row.h_out[word][3 * word] += F::ONE);
        assert!(!failures.is_empty(), "output word {word}");
    }
}

#[test]
fn ch_and_maj_are_pinned_by_their_own_constraints() {
    // Flipping `Ch` or `Maj` also breaks the addition that consumes it. A prover who adjusts
    // that addition too is stopped only by the defining constraint, so pin that one by index.
    //
    // Round `t` starts after the input and schedule constraints, with `Ch` then `Maj`.
    let round_start = |t: usize| {
        NUM_INPUT_BITS
            + SCHEDULE_EXTENSIONS * CONSTRAINTS_PER_SCHEDULE_WORD
            + t * CONSTRAINTS_PER_ROUND
    };
    for (t, bit) in [(0, 0), (40, 31), (63, 17)] {
        let failures = failures_after_edit(1, |row| row.rounds[t].ch[bit] += F::ONE);
        assert!(
            failures
                .iter()
                .any(|f| f.constraint == round_start(t) + bit),
            "Ch bit {bit} of round {t}"
        );
        let failures = failures_after_edit(2, |row| row.rounds[t].maj[bit] += F::ONE);
        assert!(
            failures
                .iter()
                .any(|f| f.constraint == round_start(t) + 32 + bit),
            "Maj bit {bit} of round {t}"
        );
    }
}

#[test]
fn rejects_non_boolean_input() {
    assert!(F::GENERATOR != F::ZERO && F::GENERATOR != F::ONE);
    // Inputs are checked first in the order a_chain[0..4], e_chain[0..4], w[0..16], so bit `b`
    // of input word `n` is constraint `32 * n + b`.
    for word in 0..24 {
        let bit = (7 * word) % 32;
        let failures = failures_after_edit(1, |row| {
            let input_word = match word {
                0..4 => &mut row.a_chain[word],
                4..8 => &mut row.e_chain[word - 4],
                _ => &mut row.w[word - 8],
            };
            input_word[bit] = F::GENERATOR;
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
    // The carry into bit 31 of a stored-carry addition, and the top bit of Maj.
    let failures = failures_after_edit(2, |row| row.rounds[9].t1_carries[0][30] = F::GENERATOR);
    assert!(!failures.is_empty());
    let failures = failures_after_edit(3, |row| row.rounds[50].maj[31] = F::GENERATOR);
    assert!(!failures.is_empty());
}

#[test]
fn rejects_message_bit_flip_without_regeneration() {
    let failures = failures_after_edit(2, |row| row.w[5][9] += F::ONE);
    assert!(!failures.is_empty());
}
