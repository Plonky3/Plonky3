//! Leaf-factor materialization benchmarks.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
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
        bencher.iter(|| {
            let mut leaves = Vec::with_capacity(height);
            for row in 0..height {
                for (value, column) in main.iter_mut().zip(&columns) {
                    *value = column[row];
                }
                leaves.push(
                    factor
                        .evaluate(BusEvaluation {
                            main: &main,
                            preprocessed: &[],
                            public: &[],
                            is_first_row: F::from_bool(row == 0),
                            is_last_row: F::from_bool(row + 1 == height),
                            is_transition: F::from_bool(row + 1 < height),
                        })
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

criterion_group!(benches, bench);
criterion_main!(benches);
