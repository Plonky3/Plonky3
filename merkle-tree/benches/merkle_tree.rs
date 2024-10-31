use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_blake3::Blake3;
use p3_commit::Mmcs;
use p3_field::{Field, PackedField, PackedValue};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_mds::integrated_coset_mds::IntegratedCosetMds;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_rescue::{BasicSboxLayer, Rescue};
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge,
    PseudoCompressionFunction, SerializingHasher32, TruncatedPermutation,
};
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;
use serde::de::DeserializeOwned;
use serde::Serialize;

fn bench_merkle_trees(criterion: &mut Criterion) {
    bench_bb_poseidon2(criterion);
    bench_bb_rescue(criterion);
    bench_bb_blake3(criterion);
    bench_bb_keccak(criterion);
}

fn bench_bb_poseidon2(criterion: &mut Criterion) {
    type F = BabyBear;

    type Perm = Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    type H = PaddingFreeSponge<Perm, 16, 8, 8>;
    let h = H::new(perm.clone());

    type C = TruncatedPermutation<Perm, 2, 8, 16>;
    let c = C::new(perm);

    bench_merkle_tree::<<F as Field>::Packing, <F as Field>::Packing, H, C, 8>(criterion, h, c);
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

    bench_merkle_tree::<<F as Field>::Packing, <F as Field>::Packing, H, C, 8>(criterion, h, c);
}

fn bench_bb_blake3(criterion: &mut Criterion) {
    type F = BabyBear;

    type H = SerializingHasher32<Blake3>;
    let h = H::new(Blake3 {});

    type C = CompressionFunctionFromHasher<Blake3, 2, 32>;
    let b = Blake3 {};
    let c = C::new(b);

    bench_merkle_tree::<F, u8, H, C, 32>(criterion, h, c);
}

fn bench_bb_keccak(criterion: &mut Criterion) {
    type F = BabyBear;

    type H = SerializingHasher32<Keccak256Hash>;
    let k = Keccak256Hash {};
    let h = H::new(k);

    type C = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
    let c = C::new(k);

    bench_merkle_tree::<F, u8, H, C, 32>(criterion, h, c);
}

fn bench_merkle_tree<P, PW, H, C, const DIGEST_ELEMS: usize>(criterion: &mut Criterion, h: H, c: C)
where
    P: PackedField,
    PW: PackedValue,
    H: CryptographicHasher<P::Scalar, [PW::Value; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
    C: Sync,
    [PW::Value; DIGEST_ELEMS]: Serialize + DeserializeOwned,
    Standard: Distribution<P::Scalar>,
{
    const ROWS: usize = 1 << 15;
    const COLS: usize = 135;

    let matrix = RowMajorMatrix::<P::Scalar>::rand(&mut thread_rng(), ROWS, COLS);
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

    let mmcs = MerkleTreeMmcs::<P, PW, H, C, DIGEST_ELEMS>::new(h, c);
    group.bench_with_input(params, &leaves, |b, input| {
        b.iter(|| mmcs.commit(input.clone()))
    });
}

criterion_group!(benches, bench_merkle_trees);
criterion_main!(benches);
