use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::Field;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

fn try_inverse(c: &mut Criterion) {
    c.bench_function("try_inverse", |b| {
        b.iter_batched(
            || rand::random::<F>(),
            |x| x.try_inverse(),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(goldilocks_arithmetic, try_inverse);
criterion_main!(goldilocks_arithmetic);
