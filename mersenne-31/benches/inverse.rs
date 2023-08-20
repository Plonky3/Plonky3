use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::Field;
use p3_mersenne_31::Mersenne31;
use rand::Rng;
type F = Mersenne31;

fn try_inverse(c: &mut Criterion) {
    c.bench_function("try_inverse", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                rng.gen::<F>()
            },
            |x| x.try_inverse(),
            BatchSize::SmallInput,
        )
    });
}
criterion_group!(mersenne31_arithmetics, try_inverse);
criterion_main!(mersenne31_arithmetics);
