#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::any::type_name;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use p3_cfft::{cfft_twiddles, CircleSubgroupFt, Radix2Cft};
use p3_field::extension::ComplexExtendable;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench_cfft(c: &mut Criterion) {
    // log_sizes correspond to the sizes of DFT we want to benchmark;
    let log_sizes = &[10, 14, 18];

    const BATCH_SIZE: usize = 1;

    test_cfft::<Mersenne31, Radix2Cft, BATCH_SIZE>(c, log_sizes);
    test_icfft::<Mersenne31, Radix2Cft, BATCH_SIZE>(c, log_sizes);
}

fn test_cfft<F, Cfft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: ComplexExtendable,
    Standard: Distribution<F>,
    Cfft: CircleSubgroupFt<F>,
{
    let mut group = c.benchmark_group(&format!(
        "cfft::<{}, {}, {}>",
        type_name::<F>(),
        type_name::<Cfft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let cfft = Cfft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &cfft, |b, dft| {
            b.iter_batched(
                || messages.clone(),
                |messages| {
                    cfft.cfft_batch(messages);
                },
                BatchSize::LargeInput,
            );
        });
    }
}

fn test_icfft<F, Cfft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: ComplexExtendable,
    Standard: Distribution<F>,
    Cfft: CircleSubgroupFt<F>,
{
    let mut group = c.benchmark_group(&format!(
        "icfft::<{}, {}, {}>",
        type_name::<F>(),
        type_name::<Cfft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let cfft = Cfft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &cfft, |b, dft| {
            b.iter_batched(
                || messages.clone(),
                |messages| {
                    cfft.icfft_batch(messages);
                },
                BatchSize::LargeInput,
            );
        });
    }
}

criterion_group!(benches, bench_cfft);
criterion_main!(benches);
