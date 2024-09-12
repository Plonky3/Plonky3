use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Bowers, Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::Complex;
use p3_field::TwoAdicField;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, Mersenne31ComplexRadix2Dit, Mersenne31Dft};
use p3_monty_31::dft::RecursiveDft;
use p3_util::pretty_name;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_fft(c: &mut Criterion) {
    // log_sizes correspond to the sizes of DFT we want to benchmark;
    // for the DFT over the quadratic extension "Mersenne31Complex" a
    // fairer comparison is to use half sizes, which is the log minus 1.
    let log_sizes = &[14, 16, 18, 20, 22];
    let log_half_sizes = &[13, 15, 17];

    const BATCH_SIZE: usize = 256;

    fft::<BabyBear, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    fft::<BabyBear, RecursiveDft<_>, BATCH_SIZE>(c, log_sizes);
    fft::<BabyBear, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    fft::<BabyBear, Radix2DitParallel, BATCH_SIZE>(c, log_sizes);
    fft::<Goldilocks, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    fft::<Goldilocks, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    fft::<Goldilocks, Radix2DitParallel, BATCH_SIZE>(c, log_sizes);
    fft::<Complex<Mersenne31>, Radix2Dit<_>, BATCH_SIZE>(c, log_half_sizes);
    fft::<Complex<Mersenne31>, Radix2Bowers, BATCH_SIZE>(c, log_half_sizes);
    fft::<Complex<Mersenne31>, Radix2DitParallel, BATCH_SIZE>(c, log_half_sizes);

    fft::<Complex<Mersenne31>, Mersenne31ComplexRadix2Dit, BATCH_SIZE>(c, log_half_sizes);
    m31_fft::<Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    m31_fft::<Mersenne31ComplexRadix2Dit, BATCH_SIZE>(c, log_sizes);

    ifft::<Goldilocks, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);

    coset_lde::<BabyBear, RecursiveDft<_>, BATCH_SIZE>(c, log_sizes);
    coset_lde::<BabyBear, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    coset_lde::<BabyBear, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    coset_lde::<BabyBear, Radix2DitParallel, BATCH_SIZE>(c, log_sizes);
    coset_lde::<Goldilocks, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
}

fn fft<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "fft/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.dft_batch(messages.clone());
            });
        });
    }
}

fn m31_fft<Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    Dft: TwoAdicSubgroupDft<Complex<Mersenne31>>,
    Standard: Distribution<Mersenne31>,
{
    let mut group = c.benchmark_group(format!(
        "m31_fft::<{}, {}>",
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                Mersenne31Dft::dft_batch::<Dft>(messages.clone());
            });
        });
    }
}

fn ifft<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "ifft/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.idft_batch(messages.clone());
            });
        });
    }
}

fn coset_lde<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "coset_lde/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.coset_lde_batch(messages.clone(), 1, F::generator());
            });
        });
    }
}

criterion_group!(benches, bench_fft);
criterion_main!(benches);
