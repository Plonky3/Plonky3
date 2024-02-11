use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::extension::Complex;
use p3_field::TwoAdicField;
use p3_fri::fold_even_odd;
use p3_goldilocks::Goldilocks;
use p3_mersenne_31::Mersenne31;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench<F: TwoAdicField>(c: &mut Criterion, log_sizes: &[usize])
where
    Standard: Distribution<F>,
{
    let name = format!("fold_even_odd::<{}>", type_name::<F>(),);
    let mut group = c.benchmark_group(&name);
    group.sample_size(10);

    for log_size in log_sizes {
        let n = 1 << log_size;

        let mut rng = thread_rng();
        let beta = rng.sample(Standard);
        let poly = rng.sample_iter(Standard).take(n).collect_vec();

        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                fold_even_odd(poly.clone(), beta);
            })
        });
    }
}

fn bench_fold_even_odd(c: &mut Criterion) {
    let log_sizes = [12, 14, 16, 18, 20, 22];

    bench::<BabyBear>(c, &log_sizes);
    bench::<Goldilocks>(c, &log_sizes);
    bench::<Complex<Mersenne31>>(c, &log_sizes);
}

criterion_group!(benches, bench_fold_even_odd);
criterion_main!(benches);
