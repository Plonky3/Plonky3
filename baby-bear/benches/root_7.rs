use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::Field;

type F = BabyBear;

fn root_7(c: &mut Criterion) {
    c.bench_function("7th_root", |b| {
        b.iter_batched(
            rand::random::<F>,
            |x| x.exp_u64(1725656503),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(baby_bear_arithmetic, root_7);
criterion_main!(baby_bear_arithmetic);
