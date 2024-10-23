use criterion::{BatchSize, BenchmarkId, Criterion};
use itertools::Itertools;
use p3_baby_bear::{BabyBear, BabyBearParameters, DiffusionMatrixBabyBear, PackedBabyBearNeon};
use p3_blake3::Blake3;
use p3_field::{AbstractField, Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::{
    pretty_hash_type, HybridMerkleTree, HybridPseudoCompressionFunction, MerkleTree,
    NodeConverter256BabyBearBytes, SimpleHybridCompressor,
};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge, SerializingHasher32,
    TruncatedPermutation,
};
use rand::{thread_rng, Rng};

type BabyBearPacking = <BabyBear as Field>::Packing;

type PermPoseidon =
    Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type HPoseidon = PaddingFreeSponge<PermPoseidon, 16, 8, 8>;
type CPoseidon = TruncatedPermutation<PermPoseidon, 2, 8, 16>;

type HBlake3 = SerializingHasher32<Blake3>;
type CBlake3 = CompressionFunctionFromHasher<Blake3, 2, 32>;

fn hybrid_compression_single() {
    let perm_poseidon = PermPoseidon::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    let h_poseidon = HPoseidon::new(perm_poseidon.clone());

    let c_poseidon = CPoseidon::new(perm_poseidon); // PseudoCompressionFunction<[BabyBear; 8]; 2]>
                                                    // [BabyBear; 8]: PackedValue<Value = [BabyBear; 8]; WIDTH = ?>
    let c_blake3 = CBlake3::new(Blake3 {});

    let c_hybrid = SimpleHybridCompressor::<_, _, _, _, 8, 32, NodeConverter256BabyBearBytes>::new(
        c_poseidon, c_blake3, false,
    );

    let dat1: [BabyBear; 10] = rand::random();
    let dat2: [BabyBear; 10] = rand::random();

    let digest_pair = [h_poseidon.hash_iter(dat1), h_poseidon.hash_iter(dat2)];

    println!(
        "Compression result (H1): {:?}",
        c_hybrid.compress(digest_pair, &[8, 4, 1], 8)
    );

    println!(
        "Compression result (H2): {:?}",
        c_hybrid.compress(digest_pair, &[8, 4, 1], 4)
    );
}

fn hybrid_compression_width<const WIDTH: usize>() {
    let perm_poseidon = PermPoseidon::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    let h_poseidon = HPoseidon::new(perm_poseidon.clone());

    let c_poseidon = CPoseidon::new(perm_poseidon);
    let c_blake3 = CBlake3::new(Blake3 {});

    let c_hybrid = SimpleHybridCompressor::<_, _, _, _, 8, 32, NodeConverter256BabyBearBytes>::new(
        c_poseidon, c_blake3, false,
    );

    let dat1: [BabyBear; 10] = rand::random();
    let dat2: [BabyBear; 10] = rand::random();

    let digest_pair = [h_poseidon.hash_iter(dat1), h_poseidon.hash_iter(dat2)];

    println!(
        "Compression result (H1): {:?}",
        c_hybrid.compress(digest_pair, &[8, 4, 1], 8)
    );

    println!(
        "Compression result (H2): {:?}",
        c_hybrid.compress(digest_pair, &[8, 4, 1], 4)
    );
}

use std::any::type_name;
use std::collections::HashMap;

fn bench_merkle_trees(criterion: &mut Criterion) {
    bench_bb_hybrid(criterion);
}

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

fn get_random_leaves_u8() -> Vec<RowMajorMatrix<u8>> {
    let mut pow2_to_size = HashMap::new();
    (0..NUM_MATRICES)
        .map(|_| {
            let mut n_rows = rand::thread_rng().gen_range(1..MAX_ROWS);
            let n_cols = rand::thread_rng().gen_range(1..MAX_COLS * 4);

            // Same-power-of-two row numbers must match
            n_rows = pow2_to_size
                .entry(n_rows.next_power_of_two())
                .or_insert(n_rows)
                .to_owned();

            RowMajorMatrix::<u8>::rand(&mut thread_rng(), n_rows, n_cols)
        })
        .collect()
}

fn bench_bb_hybrid(criterion: &mut Criterion) {
    let perm_poseidon = PermPoseidon::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    let h_poseidon = HPoseidon::new(perm_poseidon.clone());
    let h_blake3 = Blake3;

    let c_poseidon = CPoseidon::new(perm_poseidon);
    let c_blake3 = CBlake3::new(Blake3 {});
    let c_hybrid_poseidon_leaves =
        SimpleHybridCompressor::<_, _, _, _, 8, 32, NodeConverter256BabyBearBytes>::new(
            c_poseidon.clone(),
            c_blake3.clone(),
            false,
        );
    let c_hybrid_blake3_leaves = SimpleHybridCompressor::<
        CBlake3,
        CPoseidon,
        u8,
        BabyBear,
        32,
        8,
        NodeConverter256BabyBearBytes,
    >::new(c_blake3.clone(), c_poseidon, true);

    // let leaves = get_random_leaves();

    let leaves_babybear = get_random_leaves();

    let leaves_u8: Vec<RowMajorMatrix<u8>> = leaves_babybear
        .iter()
        .map(|m| {
            let new_rows = m
                .rows()
                .map(|r| {
                    let new_r: Vec<u8> = r
                        .map(|x| BabyBear::to_u32(&x).to_le_bytes())
                        .flatten()
                        .collect();

                    new_r
                })
                .flatten()
                .collect();

            RowMajorMatrix::<u8>::new(new_rows, m.width())
        })
        .collect();

    // TODO remove
    // let leaves = get_random_leaves_u8();

    let sizes = leaves_babybear
        .iter()
        .map(|m| m.rows().count())
        .sorted()
        .rev()
        .collect::<Vec<_>>();

    println!(
        "Digest: {}, Compressors: {{{}, SimpleHybridCompressor<{}, {}>>}}",
        pretty_hash_type::<HPoseidon>(),
        pretty_hash_type::<CPoseidon>(),
        pretty_hash_type::<CPoseidon>(),
        pretty_hash_type::<CBlake3>()
    );

    println!("Sizes: {:?}", sizes);
    println!(
        "BabyBear packing width: {}",
        <BabyBear as PackedValue>::WIDTH
    );
    println!(
        "BabyBearPacking packing width: {}",
        <BabyBearPacking as PackedValue>::WIDTH
    );

    let mut group = criterion.benchmark_group("MerkleTree vs HybridMerkleTree");
    group.sample_size(10);

    // group.bench_function(BenchmarkId::from_parameter("Single compressor, WIDTH = 4"), |b| {
    //     b.iter_batched(
    //         || leaves.clone(),
    //         |input| {
    //             // MerkleTree::new::<<BabyBear as Field>::Packing, _, _, _>(
    //             MerkleTree::new::<BabyBearPacking, BabyBearPacking, _, _>(
    //                 &h_poseidon,
    //                 &c_poseidon,
    //                 input,
    //             )
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    // group.bench_function(BenchmarkId::from_parameter("Single compressor, WIDTH = 1"), |b| {
    //     b.iter_batched(
    //         || leaves.clone(),
    //         |input| {
    //             MerkleTree::new::<BabyBear, BabyBear, _, _>(
    //                 &h_blake3,
    //                 &c_poseidon,
    //                 input,
    //             )
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    // group.bench_function(BenchmarkId::from_parameter("Hybrid compressor"), |b| {
    //     b.iter_batched(
    //         || leaves.clone(),
    //         |input| HybridMerkleTree::new(&h_poseidon, &c_hybrid_poseidon_leaves, input),
    //         BatchSize::SmallInput,
    //     )
    // });

    // group.bench_function(BenchmarkId::from_parameter("Single compressor"), |b| {
    //     b.iter_batched(
    //         || leaves.clone(),
    //         |input| MerkleTree::new(&h_blake3, &c_blake3, input),
    //         BatchSize::SmallInput,
    //     )
    // });

    group.bench_function(
        BenchmarkId::from_parameter("Hybrid compressor BabyBear leaves"),
        |b| {
            b.iter_batched(
                || leaves_babybear.clone(),
                |input| HybridMerkleTree::new(&h_poseidon, &c_hybrid_poseidon_leaves, input),
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_function(
        BenchmarkId::from_parameter("Hybrid compressor u8 leaves"),
        |b| {
            b.iter_batched(
                || leaves_u8.clone(),
                |input| HybridMerkleTree::new(&h_blake3, &c_hybrid_blake3_leaves, input),
                BatchSize::SmallInput,
            )
        },
    );
}

fn main() {
    // hybrid_compression_single();

    // TODO remove
    env_logger::init();

    bench_bb_hybrid(&mut Criterion::default());
}
