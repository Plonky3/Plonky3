use alloc::format;

use criterion::Criterion;
use p3_field::Field;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

pub fn benchmark_square<EF: Field>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<EF>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<EF>();
    c.bench_function(&format!("{} square", name), |b| b.iter(|| x.square()));
}

pub fn benchmark_inv<EF: Field>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<EF>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<EF>();
    c.bench_function(&format!("{} inv", name), |b| b.iter(|| x.inverse()));
}

pub fn benchmark_mul<EF: Field>(c: &mut Criterion, name: &str)
where
    Standard: Distribution<EF>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<EF>();
    let y = rng.gen::<EF>();
    c.bench_function(&format!("{} mul", name), |b| b.iter(|| x * y));
}
