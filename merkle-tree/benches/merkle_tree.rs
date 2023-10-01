use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_commit::DirectMmcs;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_mds::integrated_coset_mds::IntegratedCosetMds;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::compression::{PseudoCompressionFunction, TruncatedPermutation};
use p3_symmetric::hasher::CryptographicHasher;
use p3_symmetric::sponge::PaddingFreeSponge;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_merkle_trees(criterion: &mut Criterion) {
    bench_bb_poseidon2(criterion);
}

fn bench_bb_poseidon2(criterion: &mut Criterion) {
    type L = BabyBear;
    type D = [BabyBear; 8];

    type Mds = IntegratedCosetMds<L, 16>;
    let mds = Mds::default();

    type Perm = Poseidon2<BabyBear, Mds, DiffusionMatrixBabybear, 16, 5>;
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());

    type H = PaddingFreeSponge<BabyBear, Perm, 16, 8, 8>;
    let h = H::new(perm.clone());

    type C = TruncatedPermutation<L, Perm, 2, 8, 16>;
    let c = C::new(perm);

    bench_merkle_tree::<L, D, H, C>(criterion, h, c);
}

fn bench_merkle_tree<L, D, H, C>(criterion: &mut Criterion, h: H, c: C)
where
    L: Field,
    D: Copy + Default + PartialEq,
    H: CryptographicHasher<L, D>,
    C: PseudoCompressionFunction<D, 2>,
    Standard: Distribution<L>,
{
    const ROWS: usize = 1 << 12;
    const COLS: usize = 25;

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

    let mmcs = MerkleTreeMmcs::new(h, c);
    group.bench_with_input(params, &leaves, |b, input| {
        b.iter(|| mmcs.commit(input.clone()))
    });
}

criterion_group!(benches, bench_merkle_trees);
criterion_main!(benches);
