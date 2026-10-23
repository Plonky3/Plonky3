//! Product GKR over one tree of `2^n` leaves, and over a batched table of `2^n` leaves.
//!
//! `GF(2^192)` packs over `GF(2^64)`; the tower field `GF(2^128)` of the bus stays scalar.
//!
//! A batched table takes `n - h` batch variables with `h = ceil(0.6 n) - 1`, the split of the BitZ benchmarks.
//!
//! Run once with `RAYON_NUM_THREADS=1` for the single-thread numbers.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, BinaryField128, Poly64, Poly192};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, HashChallenger};
use p3_field::ExtensionField;
use p3_gkr::{DenseLeaves, ProductGkrProof, ProductGkrRootShape, ProductGkrShape};
use p3_keccak::Keccak256Hash;
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;

/// A Fiat-Shamir sponge over `F`.
type Challenger<F> = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// A fresh sponge.
const fn fresh<F: TranscriptField>() -> Challenger<F> {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Proves and verifies one tree of every size in `sizes`.
fn bench_field<F, EF>(c: &mut Criterion, name: &str, sizes: &[usize])
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<EF>,
    Challenger<F>: FieldChallenger<F>,
{
    let mut group = c.benchmark_group(format!("gkr/{name}"));
    group.sample_size(10);
    let mut rng = Xoroshiro128Plus::seed_from_u64(1);
    for &n in sizes {
        let shape =
            ProductGkrShape::new(n, 1, ProductGkrRootShape::Distinct).expect("a word-sized tree");
        let leaves = (0..1usize << n)
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();
        let inputs = [leaves.as_slice()];
        group.bench_function(BenchmarkId::new("prove", n), |b| {
            b.iter(|| ProductGkrProof::prove::<F, _>(&inputs, shape, &mut fresh::<F>()));
        });
        let (proof, _) = ProductGkrProof::prove::<F, _>(&inputs, shape, &mut fresh::<F>());
        group.bench_function(BenchmarkId::new("verify", n), |b| {
            b.iter(|| proof.verify::<F, _>(shape, &mut fresh::<F>()));
        });
    }
    group.finish();
}

/// Proves and verifies a batched `GF(2^192)` table of every size in `sizes`.
fn bench_batched(c: &mut Criterion, sizes: &[usize]) {
    let mut group = c.benchmark_group("gkr_batched");
    group.sample_size(10);
    let mut rng = Xoroshiro128Plus::seed_from_u64(1);
    for &n in sizes {
        let height = (3 * n).div_ceil(5) - 1;
        let shape = ProductGkrShape::batched(height, n - height).expect("a word-sized table");
        let leaves = DenseLeaves((0..1usize << n).map(|_| rng.random::<Poly192>()).collect());
        group.bench_function(BenchmarkId::new("prove", n), |b| {
            b.iter(|| {
                ProductGkrProof::prove_batched::<Poly64, _, _>(
                    &leaves,
                    shape,
                    &mut fresh::<Poly64>(),
                )
            });
        });
        let (proof, _) =
            ProductGkrProof::prove_batched::<Poly64, _, _>(&leaves, shape, &mut fresh::<Poly64>());
        group.bench_function(BenchmarkId::new("verify", n), |b| {
            b.iter(|| proof.verify::<Poly64, _>(shape, &mut fresh::<Poly64>()));
        });
    }
    group.finish();
}

fn bench_gkr(c: &mut Criterion) {
    bench_batched(c, &[16, 18, 20, 22, 24]);
    bench_field::<Poly64, Poly192>(c, "poly192", &[16, 20, 24]);
    bench_field::<BinaryField128, BinaryField128>(c, "binary128", &[16, 20, 22]);
}

criterion_group!(benches, bench_gkr);
criterion_main!(benches);
