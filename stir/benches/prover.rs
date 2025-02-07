#![cfg(feature = "test-utils")]
use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::extension::Complex;
use p3_field::TwoAdicField;
use p3_goldilocks::Goldilocks;
use p3_mersenne_31::Mersenne31;
use p3_poly::test_utils::rand_poly;
use p3_stir::{
    prover::{commit, prove},
    test_utils::{test_bb_challenger, test_bb_stir_config},
};
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit");
    group.sample_size(10);

    for log_degree in (6..=22).step_by(2) {
        let degree = 1 << log_degree;

        // Dummy values for num_rounds and log_folding factor
        let config = test_bb_stir_config(log_degree, 1, 2, 2);

        group.bench_function(BenchmarkId::from_parameter(log_degree), |b| {
            b.iter_batched(
                || rand_poly(degree - 1),
                |poly| commit(&config, poly),
                criterion::BatchSize::SmallInput,
            );
        });
    }
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove");
    group.sample_size(10);

    for log_degree in (18..=30).step_by(2) {
        let degree = 1 << log_degree;

        let log_folding_factor = 4;
        let num_rounds = log_degree / log_folding_factor - 1;
        let config = test_bb_stir_config(log_degree, 1, log_folding_factor, num_rounds);
        let challenger = test_bb_challenger();

        group.bench_function(BenchmarkId::from_parameter(log_degree), |b| {
            b.iter_batched(
                || {
                    let (witness, commitment) = commit(&config, rand_poly(degree - 1));
                    let mut challenger = challenger.clone();
                    (witness, commitment, challenger)
                },
                |(witness, commitment, mut challenger)| {
                    prove(&config, witness, commitment, &mut challenger)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
}

//criterion_group!(benches_1, bench_commit);
criterion_group!(benches_2, bench_prove);
criterion_main!(benches_2);
