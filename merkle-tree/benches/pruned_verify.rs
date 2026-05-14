//! Benchmark for amortized pruned-batch verification.
//!
//! Compares the new amortized `verify_batch_pruned` against an unamortized
//! baseline that verifies each restored authentication path independently —
//! the same work the FRI-style verifier would do today.
//!
//! FRI-shaped parameters (height 1<<20, 80 queries, binary tree) are the
//! intended consumer of this code path.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type H = PaddingFreeSponge<Perm, 16, 8, 8>;
type C = TruncatedPermutation<Perm, 2, 8, 16>;
type Mmcs2 = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, H, C, 2, 8>;

fn make_mmcs(seed: u64) -> Mmcs2 {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let h = H::new(perm.clone());
    let c = C::new(perm);
    Mmcs2::new(h, c, 0)
}

/// Pseudo-random distinct query indices, mimicking FRI challenger samples.
fn fri_indices(num_queries: usize, max_height: usize, seed: u64) -> Vec<usize> {
    use rand::RngExt;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(num_queries);
    while out.len() < num_queries {
        let i = rng.random_range(0..max_height);
        if !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

fn bench_pruned_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify_batch_pruned");
    group.sample_size(20);

    for &log_height in &[16usize, 20] {
        for &num_queries in &[40usize, 80] {
            let height = 1 << log_height;
            let cols = 4;

            let mmcs = make_mmcs(1);
            let mut rng = SmallRng::seed_from_u64(2);
            let mat = RowMajorMatrix::<F>::rand(&mut rng, height, cols);
            let dims = vec![mat.dimensions()];
            let (commit, prover_data) = mmcs.commit(vec![mat]);

            let indices = fri_indices(num_queries, height, 3);

            // Pre-compute both proof shapes so the bench measures only verify.
            let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
            let individual: Vec<_> = indices
                .iter()
                .map(|&i| mmcs.open_batch(i, &prover_data))
                .collect();

            let id = format!("h2^{log_height}_q{num_queries}");
            group.bench_with_input(
                BenchmarkId::new("amortized", &id),
                &(commit.clone(), dims.clone()),
                |b, (commit, dims)| {
                    b.iter(|| {
                        mmcs.verify_batch_pruned(commit, dims, pruned.clone())
                            .unwrap()
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("per_path", &id),
                &(commit, dims),
                |b, (commit, dims)| {
                    b.iter(|| {
                        for (idx, opening) in indices.iter().zip(&individual) {
                            let bref = BatchOpeningRef::new(
                                &opening.opened_values,
                                &opening.opening_proof,
                            );
                            mmcs.verify_batch(commit, dims, *idx, bref).unwrap();
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_pruned_verify);
criterion_main!(benches);
