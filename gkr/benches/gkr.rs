//! Batched product GKR over `GF(2^192)`, dense leaves from `2^16` to `2^24`.
//!
//! The batch takes `n - h` variables with `h = ceil(0.6 n) - 1`, the split of the BitZ benchmarks.
//!
//! Run once with `RAYON_NUM_THREADS=1` for the single-thread numbers.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_challenger::HashChallenger;
use p3_gkr::{DenseLeaves, ProductGkrProof, ProductGkrShape};
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Fiat-Shamir sponge.
type Challenger = BinaryChallenger<Poly64, HashChallenger<u8, Keccak256Hash, 32>>;

/// A fresh sponge.
const fn fresh() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

fn bench_gkr(c: &mut Criterion) {
    let mut group = c.benchmark_group("gkr_batched");
    group.sample_size(10);
    let mut rng = SmallRng::seed_from_u64(1);
    for n in [16usize, 18, 20, 22, 24] {
        let height = (3 * n).div_ceil(5) - 1;
        let shape = ProductGkrShape::batched(height, n - height).expect("a word-sized table");
        let leaves = DenseLeaves((0..1usize << n).map(|_| rng.random::<Poly192>()).collect());
        group.bench_function(BenchmarkId::new("prove", n), |b| {
            b.iter(|| ProductGkrProof::prove_batched::<Poly64, _, _>(&leaves, shape, &mut fresh()));
        });
        let (proof, _) =
            ProductGkrProof::prove_batched::<Poly64, _, _>(&leaves, shape, &mut fresh());
        group.bench_function(BenchmarkId::new("verify", n), |b| {
            b.iter(|| proof.verify::<Poly64, _>(shape, &mut fresh()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gkr);
criterion_main!(benches);
