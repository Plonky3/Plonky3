use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake3::Blake3;
use p3_symmetric::CryptographicHasher;

/// Messages per batch, large enough that per-call overhead disappears into the loop.
const MESSAGES: usize = 4096;

/// Message lengths that cover every shape the batched path distinguishes.
///
/// 32 is a digest, shorter than one block.
///
/// 64 is a two-to-one Merkle compression, exactly one block.
///
/// 256 is a wide leaf row of four whole blocks.
///
/// 540 is the leaf row of `merkle-tree`'s benchmark: whole blocks plus a short tail.
///
/// 1024 is one full chunk, and 1025 spills one byte into a second chunk.
///
/// 4096 and 8192 are multi-chunk rows, whose chunks fold into a tree.
const LENGTHS: [usize; 8] = [32, 64, 256, 540, 1024, 1025, 4096, 8192];

/// A deterministic byte stream, so every run hashes the same fixture.
fn fixture(bytes: usize) -> Vec<u8> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    (0..bytes)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

fn blake3_hash_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake3 batch of 4096 messages");

    for len in LENGTHS {
        let messages = fixture(len * MESSAGES);
        let mut digests = vec![[0u8; 32]; MESSAGES];

        group.throughput(Throughput::Bytes((len * MESSAGES) as u64));

        group.bench_with_input(BenchmarkId::new("one at a time", len), &len, |b, &len| {
            b.iter(|| {
                for (digest, message) in digests.iter_mut().zip(messages.chunks_exact(len)) {
                    *digest = Blake3.hash_slice(black_box(message));
                }
                black_box(&digests);
            });
        });

        group.bench_with_input(BenchmarkId::new("hash_many", len), &len, |b, _| {
            b.iter(|| {
                Blake3.hash_many(black_box(&messages), black_box(&mut digests));
            });
        });
    }

    group.finish();
}

/// Batches with fewer messages than lanes: a message count, then a message length.
///
/// 1, 2 and 8 long messages leave most lanes idle when each message takes one lane.
///
/// 24 messages fill a group only partly, and 8 of 4 KiB have few chunks each.
const FEW: [(usize, usize); 6] = [
    (1, 65536),
    (2, 65536),
    (8, 65536),
    (24, 65536),
    (8, 4096),
    (24, 16384),
];

fn blake3_few_long_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake3 few long messages");

    for (count, len) in FEW {
        let messages = fixture(len * count);
        let mut digests = vec![[0u8; 32]; count];
        let id = format!("{count} x {len}");

        group.throughput(Throughput::Bytes((len * count) as u64));

        group.bench_function(BenchmarkId::new("one at a time", &id), |b| {
            b.iter(|| {
                for (digest, message) in digests.iter_mut().zip(messages.chunks_exact(len)) {
                    *digest = Blake3.hash_slice(black_box(message));
                }
                black_box(&digests);
            });
        });

        group.bench_function(BenchmarkId::new("hash_many", &id), |b| {
            b.iter(|| {
                Blake3.hash_many(black_box(&messages), black_box(&mut digests));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, blake3_hash_many, blake3_few_long_messages);
criterion_main!(benches);
