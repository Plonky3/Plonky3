use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_commit::DirectMmcs;
use p3_field::{Field, PackedField};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_mds::integrated_coset_mds::IntegratedCosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_rescue::{BasicSboxLayer, Rescue};
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge,
    PseudoCompressionFunction, SerializingHasher32, TruncatedPermutation,
};
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_merkle_trees(criterion: &mut Criterion) {
    bench_bb_poseidon2(criterion);
    bench_bb_rescue(criterion);
    bench_bb_blake3(criterion);
    bench_bb_keccak(criterion);
}

fn bench_bb_poseidon2(criterion: &mut Criterion) {
    type F = BabyBear;

    type Perm = Poseidon2<BabyBear, DiffusionMatrixBabybear, 16, 7>;
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());

    type H = PaddingFreeSponge<Perm, 16, 8, 8>;
    let h = H::new(perm.clone());

    type C = TruncatedPermutation<Perm, 2, 8, 16>;
    let c = C::new(perm);

    bench_merkle_tree::<<F as Field>::Packing, H, C, 8>(criterion, h, c);
}

fn bench_bb_rescue(criterion: &mut Criterion) {
    type F = BabyBear;

    type Mds = IntegratedCosetMds<F, 16>;
    let mds = Mds::default();

    type Perm = Rescue<F, Mds, BasicSboxLayer<F>, 16>;
    let round_constants = Perm::get_round_constants_from_rng(8, &mut thread_rng());
    let perm = Perm::new(8, round_constants, mds, BasicSboxLayer::for_alpha(7));

    type H = PaddingFreeSponge<Perm, 16, 8, 8>;
    let h = H::new(perm.clone());

    type C = TruncatedPermutation<Perm, 2, 8, 16>;
    let c = C::new(perm);

    bench_merkle_tree::<<F as Field>::Packing, H, C, 8>(criterion, h, c);
}

fn bench_bb_blake3(criterion: &mut Criterion) {
    type F = BabyBear;

    type H = SerializingHasher32<Blake3>;
    let h = H::new(Blake3 {});

    type C = CompressionFunctionFromHasher<F, H, 2, 8>;
    let c = C::new(h);

    bench_merkle_tree::<F, H, C, 8>(criterion, h, c);
}

fn bench_bb_keccak(criterion: &mut Criterion) {
    type F = BabyBear;

    type H = SerializingHasher32<Keccak256Hash>;
    let h = H::new(Keccak256Hash {});

    type C = CompressionFunctionFromHasher<F, H, 2, 8>;
    let c = C::new(h);

    bench_merkle_tree::<F, H, C, 8>(criterion, h, c);
}

fn bench_merkle_tree<P, H, C, const DIGEST_ELEMS: usize>(criterion: &mut Criterion, h: H, c: C)
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
    Standard: Distribution<P::Scalar>,
{
    const ROWS: usize = 1 << 15;
    const COLS: usize = 135;

    let matrix = RowMajorMatrix::rand(&mut thread_rng(), ROWS, COLS);
    let dims = matrix.dimensions();
    let leaves = vec![matrix];

    let name = format!(
        "MerkleTree::<{}, {}>::new",
        type_name::<H>(),
        type_name::<C>()
    );
    let params = BenchmarkId::from_parameter(dims);

    let mut group = criterion.benchmark_group(name);
    group.sample_size(10);

    let mmcs = FieldMerkleTreeMmcs::<P, H, C, DIGEST_ELEMS>::new(h, c);
    group.bench_with_input(params, &leaves, |b, input| {
        b.iter(|| mmcs.commit(input.clone()))
    });
}

criterion_group!(benches, bench_merkle_trees);
criterion_main!(benches);
