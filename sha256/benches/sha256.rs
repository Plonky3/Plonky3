use core::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use p3_sha256::{Sha256, Sha256Compress};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

/// Messages per batch, large enough that per-call overhead disappears.
const BATCH: usize = 4096;

/// Message lengths that cover the shapes a Merkle tree actually feeds a hasher.
///
/// - 32 bytes is a single padded block.
/// - 64 bytes spills the padding into a second block.
/// - 256 bytes is a five-block message, where the schedule dominates.
const LENGTHS: [usize; 3] = [32, 64, 256];

/// A cheap deterministic byte stream, so every run benches the same bytes.
fn random_bytes(len: usize, mut state: u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn bench_hash_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha256 hash_many");

    for len in LENGTHS {
        let input = random_bytes(len * BATCH, 0x243f_6a88_85a3_08d3 ^ len as u64);
        let mut out = vec![[0u8; 32]; BATCH];

        group.throughput(Throughput::Bytes((len * BATCH) as u64));
        group.bench_function(format!("batched/{len}"), |b| {
            b.iter(|| Sha256.hash_many(black_box(&input), black_box(&mut out)));
        });
        group.bench_function(format!("one at a time/{len}"), |b| {
            b.iter(|| {
                for (message, digest) in black_box(&input).chunks_exact(len).zip(&mut out) {
                    *digest = Sha256.hash_slice(message);
                }
            });
        });
    }

    group.finish();
}

fn bench_compress_many(c: &mut Criterion) {
    let bytes = random_bytes(64 * BATCH, 0x13198a2e_03707344);
    let inputs: Vec<[[u8; 32]; 2]> = bytes
        .as_chunks::<64>()
        .0
        .iter()
        .map(|block| {
            [
                block[..32].try_into().unwrap(),
                block[32..].try_into().unwrap(),
            ]
        })
        .collect();
    let mut out = vec![[0u8; 32]; BATCH];

    let mut group = c.benchmark_group("sha256 compress");
    group.throughput(Throughput::Bytes((64 * BATCH) as u64));

    group.bench_function("batched", |b| {
        b.iter(|| Sha256Compress.compress_many(black_box(&inputs), black_box(&mut out)));
    });
    group.bench_function("one at a time", |b| {
        b.iter(|| {
            for (input, digest) in black_box(&inputs).iter().zip(&mut out) {
                *digest = Sha256Compress.compress(*input);
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_hash_many, bench_compress_many);
criterion_main!(benches);
