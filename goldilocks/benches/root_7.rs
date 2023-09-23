use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::AbstractField;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

fn root_7(c: &mut Criterion) {
    c.bench_function("7th_root", |b| {
        b.iter_batched(
            rand::random::<F>,
            |x| x.exp_u64(10540996611094048183),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(goldilocks_arithmetic, root_7);
criterion_main!(goldilocks_arithmetic);
