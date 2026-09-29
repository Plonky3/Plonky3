//! Commit 64 MiB matrices of narrow and wide rows with the batched byte hashers.
//! Input matrices are generated once and borrowed during each timed commitment.
//!
//! Run with a fixed worker count, for example:
//! `RAYON_NUM_THREADS=8 cargo bench -p p3-merkle-tree --features parallel --bench wide_rows`

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_field::PackedValue;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTree;
use p3_sha256::{Sha256, Sha256Compress};
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction,
    SerializingHasher,
};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

/// Bytes committed per tree.
///
/// Large enough that the input leaves the private caches of every core.
const TOTAL_BYTES: usize = 1 << 26;

/// Row sizes in bytes: a common narrow leaf and a wide one.
const ROW_BYTES: [usize; 2] = [1 << 10, 1 << 16];

fn bench_hash<F, H, C>(criterion: &mut Criterion, name: &str, h: &H, c: &C)
where
    F: PackedValue<Value = F> + Default,
    StandardUniform: Distribution<F>,
    H: CryptographicHasher<F, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    let mut group = criterion.benchmark_group(format!("wide_rows/{name}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(TOTAL_BYTES as u64));

    let mut rng = SmallRng::seed_from_u64(1);
    for row_bytes in ROW_BYTES {
        // Same byte volume for every row size, so the rows only change the shape.
        let width = row_bytes / size_of::<F>();
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, TOTAL_BYTES / row_bytes, width);

        group.bench_function(BenchmarkId::from_parameter(row_bytes), |b| {
            b.iter(|| {
                MerkleTree::<F, u8, _, 2, 32>::new::<F, u8, H, C>(h, c, vec![matrix.as_view()])
            });
        });
    }
    group.finish();
}

fn bench_wide_rows(criterion: &mut Criterion) {
    // Byte matrices hashed directly by each byte hasher.
    bench_hash::<u8, _, _>(criterion, "sha256", &Sha256, &Sha256Compress);

    let blake3 = CompressionFunctionFromHasher::<_, 2, 32>::new(Blake3);
    bench_hash::<u8, _, _>(criterion, "blake3", &Blake3, &blake3);

    let keccak = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);
    bench_hash::<u8, _, _>(criterion, "keccak", &Keccak256Hash, &keccak);

    // Field rows go through a serializing adapter before the byte hasher.
    let serializing = SerializingHasher::new(Blake3);
    bench_hash::<BabyBear, _, _>(criterion, "babybear-blake3", &serializing, &blake3);
}

criterion_group!(benches, bench_wide_rows);
criterion_main!(benches);
