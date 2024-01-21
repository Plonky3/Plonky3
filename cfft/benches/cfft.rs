#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_cfft::{cfft, cfft_inv, cfft_twiddles, CircleSubgroupFt, Radix2Cft};
use p3_field::{ComplexExtension, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, Mersenne31Complex};
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench_cfft(c: &mut Criterion) {
    // log_sizes correspond to the sizes of DFT we want to benchmark;
    let log_sizes = &[3, 6, 10];

    const BATCH_SIZE: usize = 1;

    test_cfft::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, BATCH_SIZE>(c, log_sizes);
    test_icfft::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, BATCH_SIZE>(c, log_sizes);
    cfft_timing(c, log_sizes);
    cfft_inv_timing(c, log_sizes);
}

fn cfft_timing(c: &mut Criterion, log_sizes: &[usize]) {
    let mut group = c.benchmark_group(&format!("cfft::<{}>", type_name::<Mersenne31>(),));
    group.sample_size(10);

    let mut rng = rand::thread_rng();
    for log_n in log_sizes {
        let n = 1 << log_n;

        let mut message: Vec<_> = (0..n).map(|_| rng.gen::<Mersenne31>()).collect();

        let twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(*log_n, true);

        group.bench_function(&format!("Benching Size {}", n), |b| {
            b.iter(|| cfft(&mut message, &twiddles))
        });
    }
}

fn cfft_inv_timing(c: &mut Criterion, log_sizes: &[usize]) {
    let mut group = c.benchmark_group(&format!("cfft_inv::<{}>", type_name::<Mersenne31>(),));
    group.sample_size(10);

    let mut rng = rand::thread_rng();
    for log_n in log_sizes {
        let n = 1 << log_n;

        let mut message: Vec<_> = vec![(); n]
            .iter()
            .map(|_| rng.gen::<Mersenne31>())
            .collect();

        let twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(*log_n, false);

        group.bench_function(&format!("Benching Size {}", n), |b| {
            b.iter(|| cfft_inv(&mut message, &twiddles))
        });
    }
}

fn test_cfft<Base, Ext, Cfft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    Base: Field,
    Standard: Distribution<Base>,
    Ext: ComplexExtension<Base>,
    Cfft: CircleSubgroupFt<Base, Ext>,
{
    let mut group = c.benchmark_group(&format!(
        "cfft::<{}, {}, {}>",
        type_name::<Base>(),
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
            b.iter(|| {
                cfft.cfft_batch(messages.clone());
            });
        });
    }
}

fn test_icfft<Base, Ext, Cfft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    Base: Field,
    Standard: Distribution<Base>,
    Ext: ComplexExtension<Base>,
    Cfft: CircleSubgroupFt<Base, Ext>,
{
    let mut group = c.benchmark_group(&format!(
        "icfft::<{}, {}, {}>",
        type_name::<Base>(),
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
            b.iter(|| {
                cfft.icfft_batch(messages.clone());
            });
        });
    }
}

criterion_group!(benches, bench_cfft);
criterion_main!(benches);
