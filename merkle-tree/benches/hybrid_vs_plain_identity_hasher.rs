use criterion::{BatchSize, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_merkle_tree::{
    HybridMerkleTree, MerkleTree, SimpleHybridCompressor, UnsafeNodeConverter256BabyBearBytes,
};
use p3_symmetric::IdentityHasher;

mod common;
use common::*;

type BabyBearPacking = <BabyBear as Field>::Packing;
type HIdentityBabyBear256 = IdentityHasher<8>;

// Takeaway: HybridMerkleTree with unsafe compressor is much faster than
// MerkleTree with safe compressor and WIDTH=1.

// COMMAND
// sudo cargo +nightly flamegraph --bench hybrid_vs_plain_width_1_identity_hasher --features=unsafe-conversion

const MAX_ROWS: usize = 1 << 15;
const MAX_COLS: usize = 2;
const NUM_MATRICES: usize = 8;

// TODO allow for more matrices, control only that no more than 8 are concatenated

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

    let leaves = get_random_leaves(MAX_ROWS, MAX_COLS, NUM_MATRICES);

    let mut group = criterion.benchmark_group("MerkleTree vs HybridMerkleTree with IdentityHasher");
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
