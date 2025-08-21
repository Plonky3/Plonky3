use alloc::format;
use alloc::vec::Vec;
use core::hint::black_box;

use criterion::{BatchSize, Criterion};
use p3_field::{Algebra, Field, PrimeCharacteristicRing};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Not useful for benchmarking prime fields as multiplication is too fast but
/// handy for extension fields.
pub fn benchmark_mul<F: Field>(c: &mut Criterion, name: &str)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let x = rng.random::<F>();
    let y = rng.random::<F>();
    c.bench_function(&format!("{name} mul"), |b| {
        b.iter(|| black_box(black_box(x) * black_box(y)))
    });
}

pub fn benchmark_square<F: Field>(c: &mut Criterion, name: &str)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let x = rng.random::<F>();
    c.bench_function(&format!("{name} square"), |b| {
        b.iter(|| black_box(black_box(x).square()))
    });
}

pub fn benchmark_inv<F: Field>(c: &mut Criterion, name: &str)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let x = rng.random::<F>();
    c.bench_function(&format!("{name} inv"), |b| {
        b.iter(|| black_box(black_box(x)).inverse())
    });
}

pub fn benchmark_mul_2exp<R: PrimeCharacteristicRing + Copy, const REPS: usize>(
    c: &mut Criterion,
    name: &str,
    val: u64,
) where
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mut input = Vec::new();
    for _ in 0..REPS {
        input.push(rng.random::<R>())
    }
    c.bench_function(&format!("{name} mul_2exp_u64 {val}"), |b| {
        b.iter(|| input.iter_mut().for_each(|i| *i = i.mul_2exp_u64(val)))
    });
}

pub fn benchmark_halve<F: Field, const REPS: usize>(c: &mut Criterion, name: &str)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mut input = Vec::new();
    for _ in 0..REPS {
        input.push(rng.random::<F>())
    }
    c.bench_function(&format!("{name} halve. Num Reps: {REPS}"), |b| {
        b.iter(|| input.iter_mut().for_each(|i| *i = i.halve()))
    });
}

pub fn benchmark_div_2exp<F: Field, const REPS: usize>(c: &mut Criterion, name: &str, val: u64)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mut input = Vec::new();
    for _ in 0..REPS {
        input.push(rng.random::<F>())
    }
    c.bench_function(&format!("{name} div_2exp_u64 {val}"), |b| {
        b.iter(|| input.iter_mut().for_each(|i| *i = i.div_2exp_u64(val)))
    });
}

/// Benchmark the time taken to sum an array [[F; N]; REPS] by summing each array
/// [F; N] using .sum() method and accumulating the sums into an accumulator.
///
/// Making N larger and REPS smaller (vs the opposite) leans the benchmark more sensitive towards
/// the latency (resp throughput) of the sum method.
pub fn benchmark_iter_sum<R: PrimeCharacteristicRing + Copy, const N: usize, const REPS: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mut input = Vec::new();
    for _ in 0..REPS {
        input.push(rng.random::<[R; N]>())
    }
    c.bench_function(&format!("{name} sum/{REPS}, {N}"), |b| {
        b.iter(|| {
            let mut acc = R::ZERO;
            for row in &mut input {
                acc += row.iter().copied().sum()
            }
            acc
        })
    });
}

/// Benchmark the time taken to sum an array [[F; N]; REPS] by summing each array
/// [F; N] using sum_array method and accumulating the sums into an accumulator.
///
/// Making N larger and REPS smaller (vs the opposite) leans the benchmark more sensitive towards
/// the latency (resp throughput) of the sum method.
pub fn benchmark_sum_array<R: PrimeCharacteristicRing + Copy, const N: usize, const REPS: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mut input = Vec::new();
    for _ in 0..REPS {
        input.push(rng.random::<[R; N]>())
    }
    c.bench_function(&format!("{name} tree sum/{REPS}, {N}"), |b| {
        b.iter(|| {
            let mut acc = R::ZERO;
            for row in &mut input {
                acc += R::sum_array::<N>(row)
            }
            acc
        })
    });
}

/// Benchmark the time taken to do dot products on a pair of `[R; N]` arrays.
///
/// These numbers get more trustworthy as N increases. Small N leads to the
/// computation being too fast to be measured accurately.
pub fn benchmark_dot_array<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let lhs = rng.random::<[R; N]>();
    let rhs = rng.random::<[R; N]>();

    c.bench_function(&format!("{name} dot product/{N}"), |b| {
        b.iter(|| black_box(R::dot_product(black_box(&lhs), black_box(&rhs))))
    });
}

/// Benchmark the time taken to add two slices together.
pub fn benchmark_add_slices<F: Field, const LENGTH: usize>(c: &mut Criterion, name: &str)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mut slice_1 = Vec::new();
    let mut slice_2 = Vec::new();
    for _ in 0..LENGTH {
        slice_1.push(rng.random());
        slice_2.push(rng.random());
    }
    c.bench_function(&format!("{name} add slices/{LENGTH}"), |b| {
        let mut in_slice = slice_1.clone();
        b.iter(|| {
            F::add_slices(&mut in_slice, &slice_2);
        })
    });
}

pub fn benchmark_add_latency<R: PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<R>,
{
    c.bench_function(&format!("add-latency/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
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
    c.bench_function(&format!("add-throughput/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
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
    c.bench_function(&format!("sub-latency/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
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
    c.bench_function(&format!("sub-throughput/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
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
    c.bench_function(&format!("mul-latency/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
                let mut vec = Vec::new();
                for _ in 0..N {
                    vec.push(rng.random::<R>())
                }
                vec
            },
            |x| x.iter().fold(R::ONE, |x, y| x * *y),
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
    c.bench_function(&format!("mul-throughput/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
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

pub fn benchmark_base_mul_latency<F: Field, A: Algebra<F> + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<F> + Distribution<A>,
{
    c.bench_function(&format!("base_mul-latency/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
                let mut vec = Vec::new();
                for _ in 0..N {
                    vec.push(rng.random::<F>())
                }
                let init_val = rng.random::<A>();
                (vec, init_val)
            },
            |(x, init_val)| x.iter().fold(init_val, |x, y| x * *y),
            BatchSize::SmallInput,
        )
    });
}

pub fn benchmark_base_mul_throughput<F: Field, A: Algebra<F> + Copy, const N: usize>(
    c: &mut Criterion,
    name: &str,
) where
    StandardUniform: Distribution<F> + Distribution<A>,
{
    c.bench_function(&format!("base_mul-throughput/{N} {name}"), |b| {
        b.iter_batched(
            || {
                let mut rng = SmallRng::seed_from_u64(1);
                let a_tuple = (
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                    rng.random::<A>(),
                );
                let f_tuple = (
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                    rng.random::<F>(),
                );
                (a_tuple, f_tuple)
            },
            |(
                (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j),
                (a_f, b_f, c_f, d_f, e_f, f_f, g_f, h_f, i_f, j_f),
            )| {
                for _ in 0..N {
                    (a, b, c, d, e, f, g, h, i, j) = (
                        a * a_f,
                        b * b_f,
                        c * c_f,
                        d * d_f,
                        e * e_f,
                        f * f_f,
                        g * g_f,
                        h * h_f,
                        i * i_f,
                        j * j_f,
                    );
                }
                (a, b, c, d, e, f, g, h, i, j)
            },
            BatchSize::SmallInput,
        )
    });
}
