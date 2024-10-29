use std::collections::HashMap;

use criterion::{BatchSize, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_blake3::Blake3;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{
    HybridMerkleTree, NodeConverter256BabyBearBytes, SimpleHybridCompressor,
    UnsafeNodeConverter256BabyBearBytes,
};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, TruncatedPermutation};
use rand::{thread_rng, Rng};

type PermPoseidon =
    Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type HPoseidon = PaddingFreeSponge<PermPoseidon, 16, 8, 8>;
type CPoseidon = TruncatedPermutation<PermPoseidon, 2, 8, 16>;

type CBlake3 = CompressionFunctionFromHasher<Blake3, 2, 32>;

// Takeaway: unsafe conversion is substantially faster

// sudo cargo +nightly flamegraph --bench unsafe_vs_safe --features=unsafe-conversion
// In order for the unsafe calls not to get optimised away:
// sudo cargo +nightly flamegraph --bench unsafe_vs_safe --features=unsafe-conversion --dev

const MAX_ROWS: usize = 1 << 15;
const MAX_COLS: usize = 10;
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
    // TODO remove
    env_logger::init();

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
        SimpleHybridCompressor::<_, _, _, _, 8, 32, NodeConverter256BabyBearBytes>::new(
            c_poseidon.clone(),
            c_blake3.clone(),
            false,
        );

    let c_unsafe_hybrid_poseidon_leaves =
        SimpleHybridCompressor::<_, _, _, _, 8, 32, UnsafeNodeConverter256BabyBearBytes>::new(
            c_poseidon.clone(),
            c_blake3.clone(),
            false,
        );

    let leaves = get_random_leaves();

    let mut group = criterion.benchmark_group("Safe HybridMerkleTree vs Unsafe HybridMerkleTree");
    group.sample_size(10);

    group.bench_function(BenchmarkId::from_parameter("Safe hybrid compressor"), |b| {
        b.iter_batched(
            || leaves.clone(),
            |input| HybridMerkleTree::new(&h_poseidon, &c_hybrid_poseidon_leaves, input),
            BatchSize::SmallInput,
        )
    });

    group.bench_function(
        BenchmarkId::from_parameter("Unsafe hybrid compressor"),
        |b| {
            b.iter_batched(
                || leaves.clone(),
                |input| HybridMerkleTree::new(&h_poseidon, &c_unsafe_hybrid_poseidon_leaves, input),
                BatchSize::SmallInput,
            )
        },
    );
}
