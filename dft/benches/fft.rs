use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Bowers, Radix2DFTSmallBatch, Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::{BinomialExtensionField, Complex};
use p3_field::{Algebra, BasedVectorSpace, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, Mersenne31ComplexRadix2Dit, Mersenne31Dft};
use p3_monty_31::dft::RecursiveDft;
use p3_util::pretty_name;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

fn bench_fft(c: &mut Criterion) {
    // log_sizes correspond to the sizes of DFT we want to benchmark;
    // for the DFT over the quadratic extension "Mersenne31Complex" a
    // fairer comparison is to use half sizes, which is the log minus 1.
    let log_sizes = &[14, 16, 18, 20, 22];
    let log_half_sizes = &[13, 15, 17];

    const BATCH_SIZE: usize = 256;
    type BBExt = BinomialExtensionField<BabyBear, 5>;

    fft::<BabyBear, Radix2DFTSmallBatch<_>, BATCH_SIZE>(c, log_sizes);
    fft::<BabyBear, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    fft::<BabyBear, RecursiveDft<_>, BATCH_SIZE>(c, log_sizes);
    fft::<BabyBear, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    fft::<BabyBear, Radix2DitParallel<_>, BATCH_SIZE>(c, log_sizes);
    fft::<Goldilocks, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    fft::<Goldilocks, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    fft::<Goldilocks, Radix2DitParallel<_>, BATCH_SIZE>(c, log_sizes);
    fft::<Complex<Mersenne31>, Radix2Dit<_>, BATCH_SIZE>(c, log_half_sizes);
    fft::<Complex<Mersenne31>, Radix2Bowers, BATCH_SIZE>(c, log_half_sizes);
    fft::<Complex<Mersenne31>, Radix2DitParallel<_>, BATCH_SIZE>(c, log_half_sizes);

    fft::<Complex<Mersenne31>, Mersenne31ComplexRadix2Dit, BATCH_SIZE>(c, log_half_sizes);
    m31_fft::<Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    m31_fft::<Mersenne31ComplexRadix2Dit, BATCH_SIZE>(c, log_sizes);

    ifft::<Goldilocks, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);

    coset_lde::<BabyBear, RecursiveDft<_>, BATCH_SIZE>(c, log_sizes);
    coset_lde::<BabyBear, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    coset_lde::<BabyBear, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    coset_lde::<BabyBear, Radix2DitParallel<_>, BATCH_SIZE>(c, log_sizes);
    coset_lde::<Goldilocks, Radix2Bowers, BATCH_SIZE>(c, log_sizes);

    // The FFT is much slower when handling extension fields so we use smaller sizes:
    let ext_log_sizes = &[10, 12, 14];
    const EXT_BATCH_SIZE: usize = 50;
    fft::<BBExt, Radix2Dit<_>, EXT_BATCH_SIZE>(c, ext_log_sizes);
    fft::<BBExt, Radix2DitParallel<_>, EXT_BATCH_SIZE>(c, ext_log_sizes);
    fft_algebra::<BabyBear, BBExt, Radix2DFTSmallBatch<_>, EXT_BATCH_SIZE>(c, ext_log_sizes);
    fft_algebra::<BabyBear, BBExt, Radix2Dit<_>, EXT_BATCH_SIZE>(c, ext_log_sizes);
    fft_algebra::<BabyBear, BBExt, Radix2DitParallel<_>, EXT_BATCH_SIZE>(c, ext_log_sizes);
    fft_algebra::<BabyBear, BBExt, RecursiveDft<_>, EXT_BATCH_SIZE>(c, ext_log_sizes);
}

fn fft<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "fft/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter_batched(
                || messages.clone(),
                |m| dft.dft_batch(m),
                BatchSize::LargeInput,
            );
        });
    }
}

fn fft_algebra<F, V, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    V: Algebra<F> + BasedVectorSpace<F> + Clone + Default + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<V>,
{
    let mut group = c.benchmark_group(format!(
        "fft_algebra/{}/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        pretty_name::<V>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::<V>::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter_batched(
                || messages.clone(),
                |m| dft.dft_algebra_batch(m),
                BatchSize::LargeInput,
            );
        });
    }
}

fn m31_fft<Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    Dft: TwoAdicSubgroupDft<Complex<Mersenne31>>,
    StandardUniform: Distribution<Mersenne31>,
{
    let mut group = c.benchmark_group(format!(
        "m31_fft::<{}, {}>",
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                Mersenne31Dft::dft_batch::<Dft>(&messages);
            });
        });
    }
}

fn ifft<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "ifft/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter_batched(
                || messages.clone(),
                |m| dft.idft_batch(m),
                BatchSize::LargeInput,
            );
        });
    }
}

fn coset_lde<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "coset_lde/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter_batched(
                || messages.clone(),
                |m| dft.coset_lde_batch(m, 1, F::GENERATOR),
                BatchSize::LargeInput,
            );
        });
    }
}

/// Baseline for the chunked-LDE pattern used by `Pcs::commit_quotient`.
///
/// Emulates the current PCS implementation: take a single tall matrix of
/// height `N·D`, split it row-wise into `D` matrices of height `N`, and run
/// `coset_lde_batch` separately on each.
///
/// The output is `D` matrices of height `N << added_bits`, mirroring what
/// `TwoAdicFriPcs::get_quotient_ldes` produces today.
fn coset_lde_chunked<F, Dft>(
    c: &mut Criterion,
    grid: &[(usize, usize, usize, usize)], // (log_n, log_d, ncols, added_bits)
) where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "coset_lde_chunked/{}/{}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(2);
    for &(log_n, log_d, ncols, added_bits) in grid {
        let n = 1 << log_n;
        let d = 1 << log_d;
        let nd = n * d;

        // Generate one tall matrix of height N·D, width ncols.
        let tall = RowMajorMatrix::<F>::rand(&mut rng, nd, ncols);

        let dft = Dft::default();
        let label = format!("logN={log_n}/D={d}/ncols={ncols}/B={}", 1 << added_bits);
        group.bench_with_input(BenchmarkId::from_parameter(label), &dft, |b, dft| {
            b.iter_batched(
                || tall.clone(),
                |tall_mat| {
                    // Split into D matrices of height N (mirrors `split_evals`).
                    let mut chunks: Vec<Vec<F>> =
                        (0..d).map(|_| Vec::with_capacity(n * ncols)).collect();
                    for r in 0..n {
                        for (t, chunk) in chunks.iter_mut().enumerate() {
                            let src = (r * d + t) * ncols;
                            chunk.extend_from_slice(&tall_mat.values[src..src + ncols]);
                        }
                    }

                    // Run a coset LDE on each chunk.
                    let ldes: Vec<_> = chunks
                        .into_iter()
                        .map(|values| {
                            let m = RowMajorMatrix::new(values, ncols);
                            dft.coset_lde_batch(m, added_bits, F::GENERATOR)
                        })
                        .collect();
                    ldes
                },
                BatchSize::LargeInput,
            );
        });
    }
}

fn bench_coset_lde_chunked(c: &mut Criterion) {
    // Mirror the FRI commit_quotient bench grid in the realistic `D < B`
    // regime: (log_n, log_d, ncols, added_bits).
    // `added_bits` is `log_b`; we require `added_bits > log_d`.
    // ncols matches Challenge::DIMENSION typical values (DIM=4 for BB, DIM=2 for GL).
    let grid = &[
        (16, 1, 4, 2), // logN=16, D=2, B=4
        (16, 1, 4, 3), // logN=16, D=2, B=8
        (18, 1, 4, 2), // logN=18, D=2, B=4
        (18, 1, 4, 3), // logN=18, D=2, B=8
    ];
    coset_lde_chunked::<BabyBear, Radix2DitParallel<_>>(c, grid);

    let grid_gl = &[(16, 1, 2, 2), (16, 1, 2, 3), (18, 1, 2, 2), (18, 1, 2, 3)];
    coset_lde_chunked::<Goldilocks, Radix2DitParallel<_>>(c, grid_gl);
}

criterion_group!(benches, bench_fft, bench_coset_lde_chunked);
criterion_main!(benches);
