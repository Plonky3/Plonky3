use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_fft::{Radix2BowersFft, Radix2DitFft, TwoAdicSubgroupDFT};
use p3_field::TwoAdicField;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_fft(c: &mut Criterion) {
    fft::<BabyBear, Radix2DitFft, 10>(c);
    fft::<BabyBear, Radix2BowersFft, 10>(c);
    fft::<Goldilocks, Radix2DitFft, 10>(c);
    fft::<Goldilocks, Radix2BowersFft, 10>(c);

    ifft::<Goldilocks, Radix2DitFft, 10>(c);
}

fn fft<F, DFT, const BATCH_SIZE: usize>(c: &mut Criterion)
where
    F: TwoAdicField,
    DFT: TwoAdicSubgroupDFT<F, F> + Default,
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(&format!(
        "fft::<{}, {}, {}>",
        type_name::<F>(),
        type_name::<DFT>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in [14, 16, 18, 20] {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = DFT::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.dft_batch(messages.clone());
            });
        });
    }
}

fn ifft<F, DFT, const BATCH_SIZE: usize>(c: &mut Criterion)
where
    F: TwoAdicField,
    DFT: TwoAdicSubgroupDFT<F, F> + Default,
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(&format!(
        "ifft::<{}, {}, {}>",
        type_name::<F>(),
        type_name::<DFT>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in [14, 16, 18, 20] {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = DFT::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.idft_batch(messages.clone());
            });
        });
    }
}

criterion_group!(benches, bench_fft);
criterion_main!(benches);
