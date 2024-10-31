use criterion::{BatchSize, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{
    HybridMerkleTree, MerkleTree, SimpleHybridCompressor, UnsafeNodeConverter256BabyBearBytes,
};
use p3_symmetric::IdentityHasher;
use rand::thread_rng;

mod common;
use common::*;

type BabyBearPacking = <BabyBear as Field>::Packing;
type HIdentityBabyBear256 = IdentityHasher<8>;

const ROWS: usize = 1 << 15;

fn main() {
    let mut criterion = Criterion::default();

    let h_identity = HIdentityBabyBear256 {};
    let c_poseidon = poseidon2_compressor();
    let c_blake3 = blake3_compressor();

    let c_hybrid_poseidon_leaves =
        SimpleHybridCompressor::<_, _, _, _, 8, 32, UnsafeNodeConverter256BabyBearBytes>::new(
            c_poseidon.clone(),
            c_blake3.clone(),
            false,
        );

    // Generate 8 random matrices of height ROWS and 8 random matrices of height ROWS / 2
    let leaves = (0..8)
        .map(|_| RowMajorMatrix::<BabyBear>::rand(&mut thread_rng(), ROWS, 1))
        .chain((0..8).map(|_| RowMajorMatrix::<BabyBear>::rand(&mut thread_rng(), ROWS >> 1, 1)))
        .collect::<Vec<_>>();

    let mut group = criterion.benchmark_group("MerkleTree vs HybridMerkleTree Last Two Layers");
    group.sample_size(10);

    bench_plain_merkle_tree!(
        &mut group,
        "Poseidon2 compressor, WIDTH = 1",
        &h_identity,
        &c_poseidon,
        leaves,
        1
    );

    bench_plain_merkle_tree!(
        &mut group,
        "Poseidon2 compressor, WIDTH = 4",
        &h_identity,
        &c_poseidon,
        leaves,
        4
    );

    bench_hybrid_merkle_tree!(
        &mut group,
        "Hybrid Blake3/Poseidon2 compressor, WIDTH = 1",
        &h_identity,
        &c_hybrid_poseidon_leaves,
        leaves
    );
}
