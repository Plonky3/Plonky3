use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};

type F = BabyBear;

fn try_inverse(c: &mut Criterion) {
    c.bench_function("try_inverse", |b| {
        b.iter_batched(
            || F::from_canonical_u64(rand::random::<u64>()),
            |x| x.try_inverse(),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(baby_bear_arithmetic, try_inverse);
criterion_main!(baby_bear_arithmetic);
