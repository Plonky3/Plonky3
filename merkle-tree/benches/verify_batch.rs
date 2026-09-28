//! Single openings: one node compression, and whole MMCS verifications.
//!
//! Each verification iteration checks 256 prepared openings at distinct random rows.
//!
//! Run pinned to one core, for example:
//! `taskset -c 4 cargo bench -p p3-merkle-tree --bench verify_batch`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_commit::{BatchOpening, Mmcs};
use p3_field::PackedValue;
use p3_keccak::Keccak256Hash;
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::{Sha256, Sha256Compress};
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction,
    SerializingHasher,
};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Openings verified per iteration.
const OPENINGS: usize = 256;

/// Tree heights, as log2 of the row count.
const LOG_ROWS: [u32; 3] = [10, 16, 20];

/// Bytes per row, so field rows hold 64 elements.
const ROW_BYTES: usize = 256;

fn bench_compress(c: &mut Criterion) {
    // One 64-byte node, the unit of work at every level of a path.
    let mut group = c.benchmark_group("compress");
    let pair = [[7u8; 32], [9u8; 32]];
    let blake3 = CompressionFunctionFromHasher::<_, 2, 32>::new(Blake3);
    let keccak = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);
    group.bench_function("blake3", |b| b.iter(|| blake3.compress(black_box(pair))));
    group.bench_function("keccak256", |b| b.iter(|| keccak.compress(black_box(pair))));
    group.bench_function("sha256", |b| {
        b.iter(|| Sha256Compress.compress(black_box(pair)));
    });
    group.finish();
}

/// Commit to random rows, then verify the same prepared openings every iteration.
fn bench_mmcs<F, H, C>(c: &mut Criterion, name: &str, h: H, compress: C, row_elements: usize)
where
    F: PackedValue<Value = F>,
    StandardUniform: Distribution<F>,
    H: CryptographicHasher<F, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    let mmcs = MerkleTreeMmcs::<F, u8, H, C, 2, 32>::new(h, compress, 0);
    let mut group = c.benchmark_group(format!("verify_batch/{name}"));
    group.throughput(Throughput::Elements(OPENINGS as u64));
    for log in LOG_ROWS {
        let rows = 1usize << log;
        let mut rng = SmallRng::seed_from_u64(u64::from(log));
        let values = (0..rows * row_elements)
            .map(|_| rng.sample(StandardUniform))
            .collect();
        let matrix = RowMajorMatrix::<F>::new(values, row_elements);
        let dims: [Dimensions; 1] = [p3_matrix::Matrix::dimensions(&matrix)];
        let (commit, data) = mmcs.commit(vec![matrix]);

        // Distinct rows spread over the tree, fixed by the seed.
        let queries: Vec<(usize, BatchOpening<F, _>)> = (0..OPENINGS)
            .map(|k| (k * 0x9E37_79B9 + 1) % rows)
            .map(|i| (i, mmcs.open_batch(i, &data)))
            .collect();
        drop(data);

        group.bench_function(BenchmarkId::from_parameter(format!("2^{log}")), |b| {
            b.iter(|| {
                for (index, opening) in &queries {
                    mmcs.verify_batch(black_box(&commit), &dims, *index, opening.into())
                        .unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let blake3 = CompressionFunctionFromHasher::<_, 2, 32>::new(Blake3);
    let keccak = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);

    // Byte rows, hashed directly.
    bench_mmcs::<u8, _, _>(c, "u8/blake3", Blake3, blake3.clone(), ROW_BYTES);
    bench_mmcs::<u8, _, _>(c, "u8/keccak256", Keccak256Hash, keccak, ROW_BYTES);
    bench_mmcs::<u8, _, _>(c, "u8/sha256", Sha256, Sha256Compress, ROW_BYTES);

    // Field rows, serialized to bytes first.
    bench_mmcs::<BabyBear, _, _>(
        c,
        "babybear/blake3",
        SerializingHasher::new(Blake3),
        blake3,
        ROW_BYTES / 4,
    );
}

criterion_group!(benches, bench_compress, bench_verify);
criterion_main!(benches);
