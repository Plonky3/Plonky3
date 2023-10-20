use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::{AbstractField, Field};
use p3_mersenne_31::Mersenne31;
use rand::Rng;

type F = Mersenne31;

fn bench_m31(c: &mut Criterion) {
    c.bench_function("add-latency", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                // rng.gen::<F>()
                let mut vec = Vec::new();
                for _ in 0..10000 {
                    vec.push(rng.gen::<F>())
                }
                vec
            },
            |x| x.iter().fold(F::zero(), |x, y| x + *y),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("add-throughput", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                (
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                )
            },
            |(mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j)| {
                for _ in 0..1000 {
                    (a, b, c, d, e, f, g, h, i, j) = (
                        a + b,
                        b + c,
                        c + d,
                        d + e,
                        e + f,
                        f + g,
                        g + h,
                        h + i,
                        i + j,
                        j + a,
                    );
                }
                (a, b, c, d, e, f, g, h, i, j)
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("sub-latency", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                // rng.gen::<F>()
                let mut vec = Vec::new();
                for _ in 0..10000 {
                    vec.push(rng.gen::<F>())
                }
                vec
            },
            |x| x.iter().fold(F::zero(), |x, y| x - *y),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("sub-throughput", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                (
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                    rng.gen::<F>(),
                )
            },
            |(mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j)| {
                for _ in 0..1000 {
                    (a, b, c, d, e, f, g, h, i, j) = (
                        a - b,
                        b - c,
                        c - d,
                        d - e,
                        e - f,
                        f - g,
                        g - h,
                        h - i,
                        i - j,
                        j - a,
                    );
                }
                (a, b, c, d, e, f, g, h, i, j)
            },
            BatchSize::SmallInput,
        )
    });

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

    c.bench_function("5th_root", |b| {
        b.iter_batched(
            rand::random::<F>,
            |x| x.exp_u64(1717986917),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(mersenne31_arithmetics, bench_m31);
criterion_main!(mersenne31_arithmetics);
