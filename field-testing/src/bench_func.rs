use alloc::format;
use alloc::vec::Vec;

use criterion::{black_box, BatchSize, Criterion};
use p3_field::Field;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

pub fn benchmark_square<F: Field>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<F>();
    c.bench_function(&format!("{} square", name), |b| {
        b.iter(|| black_box(black_box(x).square()))
    });
}

pub fn benchmark_inv<F: Field>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<F>();
    c.bench_function(&format!("{} inv", name), |b| {
        b.iter(|| black_box(black_box(x)).inverse())
    });
}

pub fn benchmark_mul<F: Field>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<F>();
    let y = rng.gen::<F>();
    c.bench_function(&format!("{} mul", name), |b| {
        b.iter(|| black_box(black_box(x) * black_box(y)))
    });
}

pub fn benchmark_iter_sum<F: Field, const REPS: usize, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..REPS {
        input.push(rng.gen::<[F; N]>())
    }
    let mut output = [F::zero(); REPS];
    c.bench_function(&format!("{} sum/{}, {}", name, REPS, N), |b| {
        b.iter(|| {
            for i in 0..REPS {
                output[i] = input[i].iter().cloned().sum()
            }
            output
        })
    });
}

pub fn benchmark_add_latency<F: Field, const N: usize>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    c.bench_function(&format!("{} add-latency/{}", name, N), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                let mut vec = Vec::new();
                for _ in 0..N {
                    vec.push(rng.gen::<F>())
                }
                vec
            },
            |x| x.iter().fold(F::zero(), |x, y| x + *y),
            BatchSize::SmallInput,
        )
    });
}

pub fn benchmark_add_throughput<F: Field, const N: usize>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    c.bench_function(&format!("{} add-throughput/{}", name, N), |b| {
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
                for _ in 0..N {
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
}

pub fn benchmark_sub_latency<F: Field, const N: usize>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    c.bench_function(&format!("{} sub-latency/{}", name, N), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                let mut vec = Vec::new();
                for _ in 0..N {
                    vec.push(rng.gen::<F>())
                }
                vec
            },
            |x| x.iter().fold(F::zero(), |x, y| x - *y),
            BatchSize::SmallInput,
        )
    });
}

pub fn benchmark_sub_throughput<F: Field, const N: usize>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    c.bench_function(&format!("{} sub-throughput/{}", name, N), |b| {
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
                for _ in 0..N {
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
}
