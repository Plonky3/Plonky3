use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::Field;
use p3_mersenne_31::Mersenne31;

type F = Mersenne31;

fn root_5(c: &mut Criterion) {
    c.bench_function("5th_root", |b| {
        b.iter_batched(
            rand::random::<F>,
            |x| x.exp_u64(1717986917),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(mersenne31_arithmetics, root_5);
criterion_main!(mersenne31_arithmetics);
