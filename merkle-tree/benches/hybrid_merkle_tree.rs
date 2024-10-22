use criterion::{BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, BabyBearParameters, DiffusionMatrixBabyBear, PackedBabyBearNeon};
use p3_blake3::Blake3;
use p3_field::{AbstractField, Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::{
    HybridMerkleTree, HybridPseudoCompressionFunction, MerkleTree, NodeConverter256BabyBearBytes,
    SimpleHybridCompressor,
};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge,
    PseudoCompressionFunction, SerializingHasher32, TruncatedPermutation,
};
use rand::thread_rng;

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
        c_poseidon, c_blake3,
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
        c_poseidon, c_blake3,
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

fn bench_merkle_trees(criterion: &mut Criterion) {
    bench_bb_hybrid(criterion);
}

fn bench_bb_hybrid(criterion: &mut Criterion) {
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
        c_poseidon, c_blake3,
    );

    // const ROWS: usize = 1 << 15;
    // const COLS: usize = 135;

    // TODO replace by above
    const ROWS: usize = 1 << 8;
    const COLS: usize = 135;

    let matrix = RowMajorMatrix::<BabyBear>::rand(&mut thread_rng(), ROWS, COLS);
    let dims = matrix.dimensions();
    let leaves = vec![matrix];

    let name = format!(
        "MerkleTree::<{}, SimpleHybridCompressor<{}, {}>>::new",
        type_name::<HPoseidon>(),
        type_name::<CPoseidon>(),
        type_name::<CBlake3>()
    );
    let params = BenchmarkId::from_parameter(dims);

    let mut group = criterion.benchmark_group(name);
    group.sample_size(10);

    HybridMerkleTree::new(&h_poseidon, &c_hybrid, leaves);
    // group.bench_with_input(params, &leaves, |b, input| {
    //     b.iter(|| HybridMerkleTree::new(&h_poseidon, &c_hybrid, leaves);)
    // });
}

fn main() {
    // hybrid_compression_single();

    // TODO remove
    env_logger::init();

    bench_bb_hybrid(&mut Criterion::default());
}
