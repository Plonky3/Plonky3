//! Benchmarks for [`batch_multiplicative_inverse`].
//!
//! Sizes are picked to expose the four interesting regimes:
//!
//! - **Aligned, single chunk** (e.g. 1024): every element goes through the
//!   packed (4-lane ILP) Montgomery path; baseline against which the fix
//!   must not regress.
//! - **Misaligned, single chunk** (e.g. 1023, 511): the trailing chunk has
//!   `len % 4 != 0`. Before the fix this fell entirely to the serial path;
//!   after the fix only the 1–3 leftover elements run serially.
//! - **Aligned, multi-chunk** (e.g. 1024 * k): par_chunks dispatches each
//!   chunk to packed; should show clean Rayon scaling.
//! - **Misaligned, multi-chunk** (e.g. 4099): the last chunk gets
//!   prefix-packed + tail-serial.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, batch_multiplicative_inverse};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF4 = BinomialExtensionField<F, 4>;
type EF5 = BinomialExtensionField<F, 5>;

const SIZES: &[usize] = &[
    63, 64, 65, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 1027, 4095, 4096, 4099, 16383,
    16384, 16385,
];

fn random_nonzero<G: Field>(seed: u64, n: usize) -> Vec<G>
where
    rand::distr::StandardUniform: rand::distr::Distribution<G>,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut x: G = rng.random();
            while x.is_zero() {
                x = rng.random();
            }
            x
        })
        .collect()
}

fn bench_field<G: Field>(c: &mut Criterion, group_name: &str, seed: u64)
where
    rand::distr::StandardUniform: rand::distr::Distribution<G>,
{
    let mut group = c.benchmark_group(group_name);
    for &n in SIZES {
        let xs: Vec<G> = random_nonzero(seed, n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &xs, |b, xs| {
            b.iter(|| batch_multiplicative_inverse::<G>(std::hint::black_box(xs)));
        });
    }
    group.finish();
}

fn bench_baby_bear(c: &mut Criterion) {
    bench_field::<F>(c, "batch_multiplicative_inverse/BabyBear", 0xB1);
}

fn bench_baby_bear_ext4(c: &mut Criterion) {
    bench_field::<EF4>(c, "batch_multiplicative_inverse/BabyBear-EF4", 0xB4);
}

fn bench_baby_bear_ext5(c: &mut Criterion) {
    bench_field::<EF5>(c, "batch_multiplicative_inverse/BabyBear-EF5", 0xB5);
}

criterion_group!(
    benches,
    bench_baby_bear,
    bench_baby_bear_ext4,
    bench_baby_bear_ext5
);
criterion_main!(benches);
