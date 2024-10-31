use criterion::{BatchSize, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_merkle_tree::{
    HybridMerkleTree, MerkleTree, SimpleHybridCompressor, UnsafeNodeConverter256BabyBearBytes,
};

mod common;
use common::*;

type BabyBearPacking = <BabyBear as Field>::Packing;

const MAX_ROWS: usize = 1 << 15;
const MAX_COLS: usize = 2;
const NUM_MATRICES: usize = 50;

fn main() {
    let mut criterion = Criterion::default();

    let h_poseidon = poseidon2_hasher();
    let c_poseidon = poseidon2_compressor();
    let c_blake3 = blake3_compressor();

    let c_hybrid_poseidon_leaves =
        SimpleHybridCompressor::<_, _, _, _, 8, 32, UnsafeNodeConverter256BabyBearBytes>::new(
            c_poseidon.clone(),
            c_blake3.clone(),
            false,
        );

    let leaves = get_random_leaves(NUM_MATRICES, MAX_ROWS, MAX_COLS);

    let mut group = criterion.benchmark_group("MerkleTree vs HybridMerkleTree");
    group.sample_size(10);

    bench_plain_merkle_tree!(
        &mut group,
        "Poseidon2 compressor, WIDTH = 1",
        &h_poseidon,
        &c_poseidon,
        leaves,
        1
    );

    bench_plain_merkle_tree!(
        &mut group,
        "Poseidon2 compressor, WIDTH = 4",
        &h_poseidon,
        &c_poseidon,
        leaves,
        4
    );

    bench_hybrid_merkle_tree!(
        &mut group,
        "Hybrid Blake3/Poseidon2 compressor, WIDTH = 1",
        &h_poseidon,
        &c_hybrid_poseidon_leaves,
        leaves
    );
}
