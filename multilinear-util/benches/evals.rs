use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::evals::EvaluationsList;
use p3_multilinear_util::point::MultilinearPoint;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type F = BabyBear;
type EF4 = BinomialExtensionField<F, 4>;

fn bench_eval_multilinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_multilinear");

    for num_vars in (8..=22).step_by(2) {
        let num_evals = 1 << num_vars;

        let throughput = Throughput::Bytes((num_evals * std::mem::size_of::<F>()) as u64);
        group.throughput(throughput);

        // Each benchmark is identified by the number of variables.
        let bench_id = BenchmarkId::from_parameter(num_vars);

        group.bench_with_input(bench_id, &num_vars, |b, &n_vars| {
            let mut rng = SmallRng::seed_from_u64(42);

            // Setup closure: Generate random evaluation data and a random point.
            let setup = || {
                let evals_vec: Vec<F> = (0..1 << n_vars).map(|_| rng.random()).collect();
                let evals_list = EvaluationsList::new(evals_vec);

                let point_vec: Vec<EF4> = (0..n_vars).map(|_| rng.random()).collect();
                let point = MultilinearPoint::new(point_vec);

                (evals_list, point)
            };

            let routine = |(evals_list, point): (EvaluationsList<F>, MultilinearPoint<EF4>)| {
                let _ = std::hint::black_box(evals_list.evaluate(std::hint::black_box(&point)));
            };

            b.iter_batched(setup, routine, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

criterion_group!(benches, bench_eval_multilinear);

criterion_main!(benches);
