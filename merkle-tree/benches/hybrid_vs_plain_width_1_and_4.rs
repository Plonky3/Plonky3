use std::collections::HashMap;

use criterion::{BatchSize, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_blake3::Blake3;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{
    HybridMerkleTree, MerkleTree, SimpleHybridCompressor, UnsafeNodeConverter256BabyBearBytes,
};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, TruncatedPermutation};
use rand::{thread_rng, Rng};

type BabyBearPacking = <BabyBear as Field>::Packing;

type PermPoseidon =
    Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type HPoseidon = PaddingFreeSponge<PermPoseidon, 16, 8, 8>;
type CPoseidon = TruncatedPermutation<PermPoseidon, 2, 8, 16>;

type CBlake3 = CompressionFunctionFromHasher<Blake3, 2, 32>;

// Takeaway: HybridMerkleTree with unsafe compressor is faster than MerkleTree
// with safe compressor and WIDTH=1 (but slower than WIDTH=4).

// COMMAND
// sudo cargo +nightly flamegraph --bench hybrid_vs_plain_width_1 --features=unsafe-conversion

const MAX_ROWS: usize = 1 << 15;
const MAX_COLS: usize = 2;
const NUM_MATRICES: usize = 50;

fn get_random_leaves() -> Vec<RowMajorMatrix<BabyBear>> {
    let mut pow2_to_size = HashMap::new();
    (0..NUM_MATRICES)
        .map(|_| {
            let mut n_rows = rand::thread_rng().gen_range(1..MAX_ROWS);
            let n_cols = rand::thread_rng().gen_range(1..MAX_COLS);

            // Same-power-of-two row numbers must match
            n_rows = pow2_to_size
                .entry(n_rows.next_power_of_two())
                .or_insert(n_rows)
                .to_owned();

            RowMajorMatrix::<BabyBear>::rand(&mut thread_rng(), n_rows, n_cols)
        })
        .collect()
}

fn main() {
    let mut criterion = Criterion::default();

    let perm_poseidon = PermPoseidon::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    let h_poseidon = HPoseidon::new(perm_poseidon.clone());

    let c_poseidon = CPoseidon::new(perm_poseidon);
    let c_blake3 = CBlake3::new(Blake3 {});
    let c_hybrid_poseidon_leaves =
        SimpleHybridCompressor::<_, _, _, _, 8, 32, UnsafeNodeConverter256BabyBearBytes>::new(
            c_poseidon.clone(),
            c_blake3.clone(),
            false,
        );

    let leaves = get_random_leaves();

    let mut group = criterion.benchmark_group("MerkleTree vs HybridMerkleTree");
    group.sample_size(10);

    group.bench_function(
        BenchmarkId::from_parameter("Plain compressor WIDTH = 1"),
        |b| {
            b.iter_batched(
                || leaves.clone(),
                |input| {
                    MerkleTree::new::<BabyBear, BabyBear, _, _>(&h_poseidon, &c_poseidon, input)
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_function(
        BenchmarkId::from_parameter("Plain compressor WIDTH = 4"),
        |b| {
            b.iter_batched(
                || leaves.clone(),
                |input| {
                    MerkleTree::new::<BabyBearPacking, BabyBearPacking, _, _>(
                        &h_poseidon,
                        &c_poseidon,
                        input,
                    )
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_function(BenchmarkId::from_parameter("Hybrid compressor"), |b| {
        b.iter_batched(
            || leaves.clone(),
            |input| HybridMerkleTree::new(&h_poseidon, &c_hybrid_poseidon_leaves, input),
            BatchSize::SmallInput,
        )
    });
}
