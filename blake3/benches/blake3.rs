use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake3::Blake3;
use p3_symmetric::CryptographicHasher;

/// Messages per batch, large enough that per-call overhead disappears into the loop.
const MESSAGES: usize = 4096;

/// Message lengths that cover every shape the batched path distinguishes.
///
/// 32 is a digest, below one block and therefore one compression per message, and 64 is a
/// two-to-one Merkle compression, exactly one block. 256 is a wide leaf row, four whole blocks.
///
/// 540 is the 135-column leaf row of `merkle-tree`'s benchmark: whole blocks plus a short tail,
/// which costs one extra scalar compression per message. 1024 is a full chunk, the largest
/// message the batched path accepts, and 1025 the smallest one it hands to the scalar hasher.
const LENGTHS: [usize; 6] = [32, 64, 256, 540, 1024, 1025];

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

criterion_group!(benches, blake3_hash_many);
criterion_main!(benches);
