//! Prover benchmarks for the word shift reduction.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_word::{
    AndConstraint, ConstraintSystem, Operand, Shift, ShiftKind, ShiftedValue, ValueIndex, Word64,
};
use p3_word_backend::{PackedWitness, ShiftClaim, ShiftReductionKey};

type F = BinaryField128;

/// Expands a point into Boolean equality weights.
fn equality_weights(point: &[F]) -> Vec<F> {
    // Coordinates use the word backend's most-significant-variable-first convention.
    let mut weights = vec![F::ONE];
    for &coordinate in point {
        let old_len = weights.len();
        weights.resize(2 * old_len, F::ZERO);
        for index in (0..old_len).rev() {
            let weight = weights[index];
            weights[2 * index] = weight * (F::ONE - coordinate);
            weights[2 * index + 1] = weight * coordinate;
        }
    }
    weights
}

/// Evaluates one word as a Boolean multilinear at a fixed point.
fn evaluate_word(word: Word64, weights: &[F]) -> F {
    // Only set bits contribute to the multilinear evaluation.
    let bits = word.get();
    (0..64)
        .filter(|bit| (bits >> bit) & 1 == 1)
        .map(|bit| weights[bit])
        .sum()
}

/// Builds the honest operand claim consumed outside the timed region.
fn claim(
    system: &ConstraintSystem<Word64>,
    witness: &[Word64],
    constraint_point: Vec<F>,
    bit_point: Vec<F>,
) -> ShiftClaim<F> {
    // This benchmark uses only AND relations, leaving the other families identically zero.
    let constraint_weights = equality_weights(&constraint_point);
    let bit_weights = equality_weights(&bit_point);
    let column = |select: fn(&AndConstraint<Word64>) -> &Operand<Word64>| {
        system
            .and_constraints()
            .iter()
            .enumerate()
            .map(|(row, relation)| {
                let word = select(relation).evaluate(&[], witness).unwrap();
                constraint_weights[row] * evaluate_word(word, &bit_weights)
            })
            .sum()
    };

    ShiftClaim::new(
        constraint_point,
        bit_point,
        [F::ZERO],
        [
            column(AndConstraint::left),
            column(AndConstraint::right),
            column(AndConstraint::output),
        ],
        [F::ZERO; 4],
    )
}

/// Creates a dense shifted-word statement with one relation per committed word.
fn instance(
    log_words: usize,
) -> (
    ShiftReductionKey<Word64>,
    PackedWitness<Word64>,
    ShiftClaim<F>,
) {
    // Three distinct shift spellings exercise sparse-key grouping and both proving phases.
    let count = 1usize << log_words;
    let rotate = Shift::new(ShiftKind::RotateRight, 13).unwrap();
    let arithmetic = Shift::new(ShiftKind::ArithmeticRight, 7).unwrap();
    let lane = Shift::new(ShiftKind::Lane32RotateRight, 5).unwrap();
    let constraints = (0..count)
        .map(|row| {
            let index = |offset| ValueIndex::witness((row + offset) % count).unwrap();
            AndConstraint::new(
                Operand::new(vec![
                    ShiftedValue::single(index(0), rotate),
                    ShiftedValue::plain(index(1)),
                ]),
                Operand::single(ShiftedValue::single(index(2), arithmetic)),
                Operand::single(ShiftedValue::single(index(0), lane)),
            )
        })
        .collect();
    let system = ConstraintSystem::new(0, count, vec![], constraints, vec![]).unwrap();
    let key = ShiftReductionKey::new(system).unwrap();
    let words = (0..count)
        .map(|index| Word64::new((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
        .collect::<Vec<_>>();
    let packed = PackedWitness::new(key.system(), &[], &words).unwrap();
    let constraint_point = (0..log_words)
        .map(|index| F::interpolation_node(index + 7))
        .collect();
    let bit_point = (0..6)
        .map(|index| F::interpolation_node(index + 31))
        .collect();
    let claim = claim(key.system(), &words, constraint_point, bit_point);
    (key, packed, claim)
}

/// Measures complete proving, including transcript hashing and both sumchecks.
fn shift_reduction(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("word_shift_reduction/prove");
    for log_words in [10, 14] {
        let (key, witness, claim) = instance(log_words);
        group.throughput(criterion::Throughput::Elements(1_u64 << log_words));
        group.bench_with_input(
            BenchmarkId::from_parameter(log_words),
            &log_words,
            |bencher, _| {
                bencher.iter(|| {
                    let mut challenger: BinaryChallenger<F, _> =
                        BinaryChallenger::from_hasher(Vec::new(), Keccak256Hash);
                    black_box(
                        key.prove(black_box(&witness), black_box(&claim), &mut challenger)
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(3));
    targets = shift_reduction
}
criterion_main!(benches);
