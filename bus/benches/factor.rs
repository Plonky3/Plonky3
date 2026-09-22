//! Leaf-factor materialization benchmarks.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::symbolic::{BaseEntry, SymbolicExpression, SymbolicVariable};
use p3_baby_bear::BabyBear;
use p3_bus::{
    BusActivation, BusChallenges, BusDirection, BusEvaluation, BusPlan, BusPlanInput,
    SymbolicBusInteraction,
};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

/// Trace variables and payload width of the benchmarked declaration.
const LOG_HEIGHT: usize = 20;
const WIDTH: usize = 3;

/// Doubling-chain depths standing in for a word recomposed from its bits.
const SHARED_DEPTHS: [usize; 3] = [16, 20, 24];

/// One declaration reading the leading columns of its own table.
fn declaration() -> SymbolicBusInteraction<F> {
    let fields: Vec<SymbolicExpression<F>> = (0..WIDTH)
        .map(|index| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into())
        .collect();
    SymbolicBusInteraction {
        bus_name: String::from("memory"),
        direction: BusDirection::Push,
        fields,
        activation: BusActivation::Always,
    }
}

/// One declaration whose single payload slot is a chain of `depth` shared additions.
///
/// The chain has `depth + 1` distinct nodes and two paths out of each of them.
fn shared_declaration(depth: usize) -> SymbolicBusInteraction<F> {
    let mut expression: SymbolicExpression<F> =
        SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0).into();
    for _ in 0..depth {
        expression = expression.clone() + expression;
    }
    SymbolicBusInteraction {
        bus_name: String::from("memory"),
        direction: BusDirection::Push,
        fields: vec![expression],
        activation: BusActivation::Always,
    }
}

/// Time one row of a declaration whose payload shares every arithmetic node.
///
/// A walk that revisits shared nodes costs time exponential in the depth here.
fn bench_shared(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bus-shared-subexpression");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));

    for depth in SHARED_DEPTHS {
        let interaction = shared_declaration(depth);
        let plan = BusPlan::build(&[BusPlanInput {
            log_height: 1,
            interactions: std::slice::from_ref(&interaction),
        }])
        .unwrap()
        .unwrap();
        let challenges = BusChallenges::<EF> {
            fingerprint: (0..plan.security_geometry().tuple_variables())
                .map(|index| EF::from_u64(7 + index as u64))
                .collect(),
            offset: EF::from_u64(11),
        };
        let weights = challenges.fingerprint_weights();

        group.bench_with_input(BenchmarkId::new("row", depth), &depth, |bencher, _| {
            let factor = plan
                .compile_factor(0, &interaction, &weights, challenges.offset)
                .unwrap();
            let main = [F::from_u64(5)];
            let mut workspace = Vec::new();
            bencher.iter(|| {
                black_box(
                    factor
                        .evaluate(
                            &mut workspace,
                            BusEvaluation {
                                main: &main,
                                preprocessed: &[],
                                public: &[],
                                is_first_row: F::ZERO,
                                is_last_row: F::ZERO,
                                is_transition: F::ZERO,
                            },
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

fn bench(criterion: &mut Criterion) {
    let interaction = declaration();
    let plan = BusPlan::build(&[BusPlanInput {
        log_height: LOG_HEIGHT,
        interactions: std::slice::from_ref(&interaction),
    }])
    .unwrap()
    .unwrap();
    let challenges = BusChallenges::<EF> {
        fingerprint: (0..plan.security_geometry().tuple_variables())
            .map(|index| EF::from_u64(7 + index as u64))
            .collect(),
        offset: EF::from_u64(11),
    };
    let weights = challenges.fingerprint_weights();

    let height = 1usize << LOG_HEIGHT;
    let columns: Vec<Vec<F>> = (0..WIDTH)
        .map(|column| {
            (0..height)
                .map(|row| F::from_u64((row * 7 + column) as u64))
                .collect()
        })
        .collect();

    let mut group = criterion.benchmark_group("bus-leaf-factor");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));

    // One compiled placement per declaration, resolved against base-field rows.
    group.bench_function("compiled", |bencher| {
        let factor = plan
            .compile_factor(0, &interaction, &weights, challenges.offset)
            .unwrap();
        let mut main = vec![F::ZERO; WIDTH];
        let mut workspace = Vec::new();
        bencher.iter(|| {
            let mut leaves = Vec::with_capacity(height);
            for row in 0..height {
                for (value, column) in main.iter_mut().zip(&columns) {
                    *value = column[row];
                }
                leaves.push(
                    factor
                        .evaluate(
                            &mut workspace,
                            BusEvaluation {
                                main: &main,
                                preprocessed: &[],
                                public: &[],
                                is_first_row: F::from_bool(row == 0),
                                is_last_row: F::from_bool(row + 1 == height),
                                is_transition: F::from_bool(row + 1 < height),
                            },
                        )
                        .unwrap(),
                );
            }
            black_box(leaves)
        });
    });

    // One placement per row, resolved against rows already lifted to the challenge field.
    group.bench_function("per-row", |bencher| {
        let mut main = vec![EF::ZERO; WIDTH];
        bencher.iter(|| {
            let mut leaves = Vec::with_capacity(height);
            for row in 0..height {
                for (value, column) in main.iter_mut().zip(&columns) {
                    *value = column[row].into();
                }
                leaves.push(
                    plan.evaluate_factor(
                        0,
                        &interaction,
                        BusEvaluation {
                            main: &main,
                            preprocessed: &[],
                            public: &[],
                            is_first_row: EF::from_bool(row == 0),
                            is_last_row: EF::from_bool(row + 1 == height),
                            is_transition: EF::from_bool(row + 1 < height),
                        },
                        &weights,
                        challenges.offset,
                    )
                    .unwrap(),
                );
            }
            black_box(leaves)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_shared, bench);
criterion_main!(benches);
