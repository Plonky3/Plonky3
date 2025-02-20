use alloc::format;
use alloc::vec::Vec;

use criterion::{black_box, BatchSize, Criterion};
use p3_field::{Field, PrimeCharacteristicRing};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::Rng;

pub fn benchmark_square<F: Field>(c: &mut Criterion, name: &str)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    let x = rng.random::<F>();
    c.bench_function(&format!("{} square", name), |b| {
        b.iter(|| black_box(black_box(x).square()))
    });
}

pub fn benchmark_inv<F: Field>(c: &mut Criterion, name: &str)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    let x = rng.random::<F>();
    c.bench_function(&format!("{} inv", name), |b| {
        b.iter(|| black_box(black_box(x)).inverse())
    });
}

/// Benchmark the time taken to sum an array [F; N] using .sum() method.
/// Repeat the summation REPS times.
pub fn benchmark_iter_sum<F: Field, const N: usize, const REPS: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    let mut input = Vec::new();
    for _ in 0..REPS {
        input.push(rng.random::<[F; N]>())
    }
    let mut output = [F::ZERO; REPS];
    c.bench_function(&format!("{} sum/{}, {}", name, REPS, N), |b| {
        b.iter(|| {
            for i in 0..REPS {
                output[i] = input[i].iter().copied().sum()
            }
            output
        })
    });
}

pub fn benchmark_add_latency<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    c.bench_function(&format!("add-latency/{} {}", N, name), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                let mut vec = Vec::new();
                for _ in 0..N {
                    vec.push(rng.random::<R>())
                }
                vec
            },
            |x| x.iter().fold(R::ZERO, |x, y| x + *y),
            BatchSize::SmallInput,
        )
    });
}

pub fn benchmark_add_throughput<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    c.bench_function(&format!("add-throughput/{} {}", N, name), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                (
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
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

pub fn benchmark_sub_latency<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    c.bench_function(&format!("sub-latency/{} {}", N, name), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                let mut vec = Vec::new();
                for _ in 0..N {
                    vec.push(rng.random::<R>())
                }
                vec
            },
            |x| x.iter().fold(R::ZERO, |x, y| x - *y),
            BatchSize::SmallInput,
        )
    });
}

pub fn benchmark_sub_throughput<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    c.bench_function(&format!("sub-throughput/{} {}", N, name), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                (
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
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

pub fn benchmark_mul_latency<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    c.bench_function(&format!("mul-latency/{} {}", N, name), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                let mut vec = Vec::new();
                for _ in 0..N {
                    vec.push(rng.random::<R>())
                }
                vec
            },
            |x| x.iter().fold(R::ZERO, |x, y| x * *y),
            BatchSize::SmallInput,
        )
    });
}

pub fn benchmark_mul_throughput<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    c.bench_function(&format!("mul-throughput/{} {}", N, name), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                (
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                    rng.random::<R>(),
                )
            },
            |(mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j)| {
                for _ in 0..N {
                    (a, b, c, d, e, f, g, h, i, j) = (
                        a * b,
                        b * c,
                        c * d,
                        d * e,
                        e * f,
                        f * g,
                        g * h,
                        h * i,
                        i * j,
                        j * a,
                    );
                }
                (a, b, c, d, e, f, g, h, i, j)
            },
            BatchSize::SmallInput,
        )
    });
}
