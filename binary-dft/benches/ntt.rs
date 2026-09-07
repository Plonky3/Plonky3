//! Additive NTT and Reed–Solomon encoder benchmarks.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_binary_dft::{AdditiveNtt, AdditiveRsEncoder, LchNtt, PolyBasisNtt};
use p3_binary_field::{BinaryField32, BinaryField64, BinaryField128, Ghash128, TowerLevel};
use p3_commit::Encoder;
use p3_dft::Radix2DFTSmallBatch;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

/// WHIR's folding-block width.
const WIDTH: usize = 16;

/// The base-two logarithms of the transform heights the sweep covers.
const LOG_HEIGHTS: [usize; 4] = [14, 16, 18, 20];

/// One added bit of domain, so the codewords have rate `1/2`.
const LOG_INV_RATE: usize = 1;

/// A `BinaryField128` symbol is 16 bytes and a `BabyBear` symbol 4, so equal byte volume is
/// four times as many rows.
const BABY_BEAR_ROW_RATIO: usize = 4;

fn bench_ntt(c: &mut Criterion) {
    ntt(c, "32", &LchNtt::<BinaryField32>::default());
    ntt(c, "64", &LchNtt::<BinaryField64>::default());

    // The three routes to a `GF(2^128)` transform, in the order they cost.
    //
    //     128/tower  every twiddle multiply changes basis twice and back
    //     128/hybrid the matrix changes basis once on the way in and once on the way out
    //     128/ghash  the data is already in the basis the multiply wants
    ntt(c, "128/tower", &LchNtt::<BinaryField128>::default());
    ntt(c, "128/hybrid", &PolyBasisNtt::default());
    ntt(c, "128/ghash", &LchNtt::<Ghash128>::default());
}

/// The forward transform of a width-[`WIDTH`] matrix over `S_ℓ`, across [`LOG_HEIGHTS`].
fn ntt<F: TowerLevel, N: AdditiveNtt<F>>(c: &mut Criterion, name: &str, ntt: &N)
where
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("ntt/{name}"));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for log_height in LOG_HEIGHTS {
        let coeffs = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_height, WIDTH);
        group.bench_with_input(BenchmarkId::from_parameter(log_height), ntt, |b, ntt| {
            b.iter_batched(
                || coeffs.clone(),
                |m| ntt.ntt_batch(m),
                BatchSize::PerIteration,
            );
        });
    }
}

/// Reed–Solomon encoding at equal byte volume: both arms carry `2^log_height · WIDTH · 16`
/// bytes of message, so the parameter of each pair of entries is the `BinaryField128` height
/// and the `BabyBear` matrix is [`BABY_BEAR_ROW_RATIO`] times taller.
fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    let binary = AdditiveRsEncoder::<BinaryField128>::default();
    let baby_bear = Radix2DFTSmallBatch::<BabyBear>::default();

    for log_height in LOG_HEIGHTS {
        let message = RowMajorMatrix::<BinaryField128>::rand(&mut rng, 1 << log_height, WIDTH);
        group.bench_with_input(
            BenchmarkId::new("BinaryField128", log_height),
            &binary,
            |b, encoder| {
                b.iter_batched(
                    || message.clone(),
                    |m| encoder.encode_batch(m, LOG_INV_RATE),
                    BatchSize::PerIteration,
                );
            },
        );

        let message =
            RowMajorMatrix::<BabyBear>::rand(&mut rng, BABY_BEAR_ROW_RATIO << log_height, WIDTH);
        group.bench_with_input(
            BenchmarkId::new("BabyBear", log_height),
            &baby_bear,
            |b, encoder| {
                b.iter_batched(
                    || message.clone(),
                    |m| encoder.encode_batch(m, LOG_INV_RATE),
                    BatchSize::PerIteration,
                );
            },
        );
    }
}

criterion_group!(benches, bench_ntt, bench_encode);
criterion_main!(benches);
