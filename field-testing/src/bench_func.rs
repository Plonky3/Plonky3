use alloc::vec::Vec;
use alloc::format;

use criterion::{black_box, Criterion};
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

pub fn benchmark_sum<F: Field, const N: usize>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..N {
        input.push(rng.gen::<F>())
    }

    c.bench_function(&format!("{} sum/{}", name, N), |b| {
        b.iter(|| black_box(input.iter().cloned().sum::<F>()))
    });
}
