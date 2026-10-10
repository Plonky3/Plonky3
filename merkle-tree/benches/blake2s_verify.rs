//! Verifying Merkle openings with BLAKE2s leaves and nodes.
//!
//! A verifier hashes one opened row, then one node per level of the path.
//!
//! Every one of those is a single BLAKE2s message, so this measures the one-message path end to end.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake2s::Blake2s256;
use p3_commit::Mmcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::CompressionFunctionFromHasher;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The tree's node hash: BLAKE2s of the two 32-byte children.
type Node = CompressionFunctionFromHasher<Blake2s256, 2, 32>;

/// Plonky3's Merkle commitment over byte rows, with BLAKE2s leaves and nodes.
type Blake2sMmcs = MerkleTreeMmcs<u8, u8, Blake2s256, Node, 2, 32>;

/// Openings verified per iteration.
const OPENINGS: usize = 64;

/// Trees as (log2 rows, row bytes).
const SHAPES: [(u32, usize); 2] = [(10, 256), (20, 256)];

fn verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake2s MMCS verify");
    group.throughput(Throughput::Elements(OPENINGS as u64));

    for (log, width) in SHAPES {
        let rows = 1usize << log;
        let mut rng = SmallRng::seed_from_u64(u64::from(log));
        let matrix = RowMajorMatrix::new((0..rows * width).map(|_| rng.random()).collect(), width);
        let dims = [p3_matrix::Matrix::dimensions(&matrix)];

        let mmcs = Blake2sMmcs::new(Blake2s256, Node::new(Blake2s256), 0);
        let (commit, data) = mmcs.commit(vec![matrix]);

        // The same random positions for every run.
        let queries: Vec<_> = (0..OPENINGS)
            .map(|_| {
                let index = rng.random_range(0..rows);
                (index, mmcs.open_batch(index, &data))
            })
            .collect();

        group.bench_function(
            BenchmarkId::new("64 openings", format!("2^{log} rows of {width} B")),
            |b| {
                b.iter(|| {
                    for (index, opening) in &queries {
                        mmcs.verify_batch(black_box(&commit), &dims, *index, opening.into())
                            .expect("an honest opening verifies");
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, verify);
criterion_main!(benches);
