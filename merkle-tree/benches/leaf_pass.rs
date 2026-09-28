//! Build byte-matrix trees whose rows are wide and whose leaf count is small.
//!
//! A few hundred rows of tens of kilobytes is the shape where the leaf layer is nearly all the work.
//!
//! Run with the worker count fixed, for example:
//! `RAYON_NUM_THREADS=32 cargo bench -p p3-merkle-tree --features parallel --bench leaf_pass`

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake3::Blake3;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrixView;
use p3_merkle_tree::MerkleTree;
use p3_sha256::{Sha256, Sha256Compress};
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Tree shapes as (rows, bytes per row), each committing 16 MiB.
///
/// - 256 rows of 64 KiB: fewer leaves than a wide vector group times the worker count.
/// - 4096 rows of 4 KiB: enough leaves to split into many blocks.
const SHAPES: [(usize, usize); 2] = [(256, 64 * 1024), (4096, 4 * 1024)];

fn bench_hash<H, C>(criterion: &mut Criterion, name: &str, h: &H, c: &C, input: &[u8])
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    let mut group = criterion.benchmark_group(format!("leaf_pass/{name}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(input.len() as u64));

    for (rows, row_bytes) in SHAPES {
        // The matrix borrows the shared input, so timing covers the tree build alone.
        let bytes = &input[..rows * row_bytes];
        group.bench_function(
            BenchmarkId::from_parameter(format!("{rows}x{row_bytes}")),
            |b| {
                b.iter(|| {
                    let matrix = RowMajorMatrixView::new(bytes, row_bytes);
                    let tree =
                        MerkleTree::<u8, u8, _, 2, 32>::new::<u8, u8, _, _>(h, c, vec![matrix]);
                    black_box(tree.root());
                });
            },
        );
    }
    group.finish();
}

fn bench_leaf_pass(criterion: &mut Criterion) {
    // One random input, sliced by every shape.
    let len = SHAPES
        .iter()
        .map(|(rows, bytes)| rows * bytes)
        .max()
        .unwrap();
    let mut rng = SmallRng::seed_from_u64(1);
    let input: Vec<u8> = (0..len).map(|_| rng.random()).collect();

    bench_hash(
        criterion,
        "blake3",
        &Blake3,
        &CompressionFunctionFromHasher::<_, 2, 32>::new(Blake3),
        &input,
    );
    bench_hash(criterion, "sha256", &Sha256, &Sha256Compress, &input);
    bench_hash(
        criterion,
        "keccak256",
        &Keccak256Hash,
        &CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash),
        &input,
    );
}

criterion_group!(benches, bench_leaf_pass);
criterion_main!(benches);
