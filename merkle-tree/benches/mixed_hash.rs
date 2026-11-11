//! Mixed-hash Merkle trees: a byte hash near the leaves, Poseidon2 near the root.
//!
//! Every tree hashes its leaves with the same Poseidon2 sponge, so only the compression differs.
//! `upper_from = k` compresses levels `0 .. k` with the byte hash and the rest with Poseidon2:
//!
//! - `poseidon2`: Poseidon2 everywhere, the recursion-friendly baseline.
//! - `<byte hash>/k`: the mixed trees. A Merkle path checks `k` byte compressions.
//! - `<byte hash>`: the byte hash everywhere, the native-speed bound.
//!
//! Narrow rows make compression a large share of the work; wide rows leave it to leaf hashing.
//!
//! The intended invocation is `RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-merkle-tree --features parallel --bench mixed_hash`.
//! Without `-Ctarget-cpu=native` there is no SIMD packing, so Poseidon2 runs one lane at a time and the results are not comparable.
//! Rows a few percent apart are within run-to-run noise.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_blake3::Blake3;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTree;
use p3_sha256::Sha256Compress;
use p3_symmetric::{
    CompressionFunctionFromHasher, MixedCompression, PaddingFreeSponge, PseudoCompressionFunction,
    SerializingCompression, TruncatedPermutation,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type P = <F as Field>::Packing;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Poseidon2Compress = TruncatedPermutation<Perm, 2, 8, 16>;

const LOG_ROWS: usize = 18;
const WIDTHS: [usize; 2] = [8, 64];
const UPPER_FROM: [usize; 3] = [4, 8, 12];

fn bench_mixed_hash(criterion: &mut Criterion) {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let h = Hash::new(perm.clone());
    let poseidon2 = Poseidon2Compress::new(perm);
    let sha256 = SerializingCompression::new(Sha256Compress);
    let blake3 =
        SerializingCompression::new(CompressionFunctionFromHasher::<_, 2, 32>::new(Blake3));

    for width in WIDTHS {
        let mut group = criterion.benchmark_group(format!("mixed_hash/2^{LOG_ROWS}x{width}"));
        group.sample_size(10);
        group.throughput(Throughput::Elements(1 << LOG_ROWS));
        let matrix = RowMajorMatrix::<F>::rand(
            &mut SmallRng::seed_from_u64(width as u64),
            1 << LOG_ROWS,
            width,
        );

        bench_tree(&mut group, "poseidon2", &h, &poseidon2, &matrix);
        bench_lower(&mut group, "sha256", &h, &poseidon2, &sha256, &matrix);
        bench_lower(&mut group, "blake3", &h, &poseidon2, &blake3, &matrix);
        group.finish();
    }
}

/// The mixed trees over one byte hash, then that byte hash alone.
fn bench_lower<L>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    h: &Hash,
    upper: &Poseidon2Compress,
    lower: &L,
    matrix: &RowMajorMatrix<F>,
) where
    L: PseudoCompressionFunction<[F; 8], 2> + PseudoCompressionFunction<[P; 8], 2> + Sync,
{
    for upper_from in UPPER_FROM {
        let mixed = MixedCompression::new(lower.clone(), upper.clone(), upper_from);
        bench_tree(group, &format!("{name}/{upper_from}"), h, &mixed, matrix);
    }
    bench_tree(group, name, h, lower, matrix);
}

fn bench_tree<C>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    h: &Hash,
    c: &C,
    matrix: &RowMajorMatrix<F>,
) where
    C: PseudoCompressionFunction<[F; 8], 2> + PseudoCompressionFunction<[P; 8], 2> + Sync,
{
    group.bench_function(BenchmarkId::from_parameter(name), |b| {
        b.iter_batched(
            || vec![matrix.clone()],
            |leaves| MerkleTree::<F, F, _, 2, 8>::new::<P, P, _, _>(h, c, leaves),
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, bench_mixed_hash);
criterion_main!(benches);
