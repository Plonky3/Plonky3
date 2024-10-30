use std::collections::HashMap;

use criterion::{BatchSize, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_blake3::Blake3;
use p3_field::{Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{
    Blake3Wide, HybridMerkleTree, HybridPseudoCompressionFunction, MerkleTree,
    SimpleHybridCompressor, UnsafeNodeConverter256BabyBearBytes,
};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, IdentityHasher, TruncatedPermutation,
};
use rand::{thread_rng, Rng};

type BabyBearPacking = <BabyBear as Field>::Packing;
const WIDTH: usize = BabyBearPacking::WIDTH;

type PermPoseidon =
    Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type HIdentityBabyBear256 = IdentityHasher<8>;
type CPoseidon = TruncatedPermutation<PermPoseidon, 2, 8, 16>;

type CBlake3 = CompressionFunctionFromHasher<Blake3Wide<WIDTH>, 2, 32>;

// Takeaway: HybridMerkleTree with unsafe compressor is much faster than
// MerkleTree with safe compressor and WIDTH=1.

// COMMAND
// sudo cargo +nightly flamegraph --bench hybrid_vs_plain_width_1_identity_hasher --features=unsafe-conversion

const MAX_ROWS: usize = 1 << 15;
const MAX_COLS: usize = 2;
const NUM_MATRICES: usize = 8;

// TODO allow for more matrices, control only that no more than 8 are concatenated

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

    let h_identity = HIdentityBabyBear256 {};

    let c_poseidon = CPoseidon::new(perm_poseidon);
    let c_blake3 = CBlake3::new(Blake3Wide {});

    // let c_hybrid =
    //     SimpleHybridCompressor::<
    //         CPoseidon,
    //         CBlake3,
    //         BabyBearPacking,
    //         [u8; WIDTH],
    //         8,
    //         32,
    //         UnsafeNodeConverter256BabyBearBytes
    //     >::new(
    //         c_poseidon.clone(),
    //         c_blake3.clone(),
    //         false,
    //     );

    let c_hybrid = SimpleHybridCompressor::<
        CPoseidon,
        CBlake3,
        BabyBear,
        u8,
        8,
        32,
        UnsafeNodeConverter256BabyBearBytes,
    >::new(c_poseidon.clone(), c_blake3.clone(), false);

    // c_hybrid.compress([[BabyBearPacking::default(); 8]; 2], &[16, 0], 0);
    c_hybrid.compress([[BabyBear::default(); 8]; 2], &[16, 0], 0);

    // the trait bound `UnsafeNodeConverter256BabyBearBytes:
    // NodeConverter<BabyBearNeon; 8], [[u8; 4]; 32]>` is not satisfied
    // <
    //     SimpleHybridCompressor::<
    //         CPoseidon,
    //         CBlake3,
    //         BabyBearPacking,
    //         [u8; WIDTH],
    //         8,
    //         32,
    //         UnsafeNodeConverter256BabyBearBytes
    //     > as
    //     HybridPseudoCompressionFunction<[BabyBearPacking; 8], 2>
    // >::compress(&c_hybrid, [[BabyBearPacking::default(); 8]; 2], &[16, 0], 0);

    let leaves = get_random_leaves();

    let mut group = criterion.benchmark_group("MerkleTree vs HybridMerkleTree with IdentityHasher");
    group.sample_size(10);

    group.bench_function(
        BenchmarkId::from_parameter("Poseidon2 compressor, WIDTH = 4"),
        |b| {
            b.iter_batched(
                || leaves.clone(),
                |input| {
                    MerkleTree::new::<BabyBearPacking, BabyBearPacking, _, _>(
                        &h_identity,
                        &c_poseidon,
                        input,
                    )
                },
                BatchSize::SmallInput,
            )
        },
    );

    // `SimpleHybridCompressor<TruncatedPermutation<Poseidon2<p3_monty_31::monty_31::MontyField31<BabyBearParameters>,
    // Poseidon2ExternalMatrixGeneral,
    // p3_monty_31::poseidon2::DiffusionMatrixMontyField31<BabyBearDiffusionMatrixParameters>,
    // 16, 7>, 2, 8, 16>, CompressionFunctionFromHasher<Blake3Wide<4>, 2, 32>,
    // p3_monty_31::monty_31::MontyField31<BabyBearParameters>, u8, 8, 32,
    // UnsafeNodeConverter256BabyBearBytes>:
    // HybridPseudoCompressionFunction<[p3_monty_31::aarch64_neon::packing::PackedMontyField31Neon<BabyBearParameters>;
    // 8], 2>` is not satisfied

    // `SimpleHybridCompressor<TruncatedPermutation<Poseidon2<p3_monty_31::monty_31::MontyField31<BabyBearParameters>,
    // Poseidon2ExternalMatrixGeneral,
    // p3_monty_31::poseidon2::DiffusionMatrixMontyField31<BabyBearDiffusionMatrixParameters>,
    // 16, 7>, 2, 8, 16>, CompressionFunctionFromHasher<Blake3Wide<4>, 2, 32>,
    // p3_monty_31::aarch64_neon::packing::PackedMontyField31Neon<BabyBearParameters>,
    // [u8; 4], 8, 32, UnsafeNodeConverter256BabyBearBytes>:
    // HybridPseudoCompressionFunction<[p3_monty_31::monty_31::MontyField31<BabyBearParameters>;
    // 8], 2>`

    // `SimpleHybridCompressor<
    //    CPoseidon,
    //    CBlake3
    //    NeonBabyBear,
    //    [u8; 4],
    //    8,
    //    32,
    //    UnsafeNodeConverter256BabyBearBytes>:
    //
    // HybridPseudoCompressionFunction<[BabyBear; 8], 2>`

    // SimpleHybridCompressor<
    //      C1: Poseidon2,
    //      C2: CompressionFunctionFromHasher<Blake3Wide<4>, 2, 32>,
    //      W1: BabyBearNeon,
    //      W2: u8,
    //      DIGEST_ELEMS_1: 8,
    //      DIGEST_ELEMS_2: 32,
    //      NC: UnsafeNodeConverter256BabyBearBytes>:
    //
    // HybridPseudoCompressionFunction<[BabyBearNeon; 8], 2>` is not satisfied

    // pub struct SimpleHybridCompressor<
    // C1,
    // C2,
    // W1,
    // W2,
    // const DIGEST_ELEMS_1: usize,
    // const DIGEST_ELEMS_2: usize,
    // NC,

    // group.bench_function(
    //     BenchmarkId::from_parameter("Hybrid Blake3/{Poseidon2, WIDTH = 4} compressor"),
    //     |b| {
    //         b.iter_batched(
    //             || leaves.clone(),
    //             |input| HybridMerkleTree::new::<BabyBearPacking, BabyBearPacking, _, _>(&h_identity, &c_hybrid, input),
    //             BatchSize::SmallInput,
    //         )
    //     },
    // );

    let a = <HIdentityBabyBear256 as CryptographicHasher<BabyBearPacking, [BabyBearPacking; 8]>>::hash_iter(&h_identity, [BabyBearPacking::default(); 8]);
}
