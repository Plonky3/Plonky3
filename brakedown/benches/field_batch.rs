use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_field::mersenne31::Mersenne31;
use p3_field::Field;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};
use std::any::type_name;

fn bench_field_batch(c: &mut Criterion) {
    add_scaled_slice_in_place::<Mersenne31>(c);
}

fn add_scaled_slice_in_place<F: Field>(c: &mut Criterion)
where
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(&format!(
        "add_scaled_slice_in_place::<{}>",
        type_name::<F>()
    ));
    group.sample_size(10);
    let mut rng = thread_rng();

    for n_log in [12, 24] {
        let n = 1 << n_log;
        let mut xs: Vec<F> = (&mut rng).sample_iter(Standard).take(n).collect();
        let ys: Vec<F> = (&mut rng).sample_iter(Standard).take(n).collect();
        let s: F = rng.gen();
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| F::add_scaled_slice_in_place(&mut xs, &ys, s));
        });
    }
}

criterion_group!(benches, bench_field_batch);
criterion_main!(benches);
