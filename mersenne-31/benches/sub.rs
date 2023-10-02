use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::AbstractField;
use p3_mersenne_31::Mersenne31;

use rand::Rng;

type F = Mersenne31;

fn sub(c: &mut Criterion) {
    c.bench_function("sub", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                // rng.gen::<F>()
                let mut vec = Vec::new();
                for _ in 0..10000 {
                    vec.push(rng.gen::<F>())
                }
                vec
                // vec
            },
            |x| x.iter().fold(F::ZERO, |x, y| x + *y),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(mersenne31_arithmetics, sub);
criterion_main!(mersenne31_arithmetics);