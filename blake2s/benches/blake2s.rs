use core::hint::black_box;

use blake2::digest::consts::U32;
use blake2::{Blake2s, Digest};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_blake2s::Blake2s256;
use p3_symmetric::CryptographicHasher;

/// Messages per batch, large enough that per-call overhead disappears into the loop.
const MESSAGES: usize = 4096;

/// Message lengths that cover every shape the batched path distinguishes.
///
/// 32 is a digest and 64 exactly one block, which is the two-to-one Merkle compression. 256
/// is a wide leaf row at four blocks, 540 a leaf row with a short tail, and 1024 a long
/// message where the per-call cost has long since stopped mattering.
const LENGTHS: [usize; 5] = [32, 64, 256, 540, 1024];

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

fn blake2s_hash_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake2s batch of 4096 messages");

    for len in LENGTHS {
        let messages = fixture(len * MESSAGES);
        let mut digests = vec![[0u8; 32]; MESSAGES];

        group.throughput(Throughput::Bytes((len * MESSAGES) as u64));

        group.bench_with_input(
            BenchmarkId::new("this crate, one at a time", len),
            &len,
            |b, &len| {
                b.iter(|| {
                    for (digest, message) in digests.iter_mut().zip(messages.chunks_exact(len)) {
                        *digest = Blake2s256::hash(black_box(message));
                    }
                    black_box(&digests);
                });
            },
        );

        // The reference every claim here is measured against, so a batch is never compared
        // only against this crate's own scalar path.
        group.bench_with_input(
            BenchmarkId::new("blake2 crate directly", len),
            &len,
            |b, &len| {
                b.iter(|| {
                    for (digest, message) in digests.iter_mut().zip(messages.chunks_exact(len)) {
                        *digest = Blake2s::<U32>::digest(black_box(message)).into();
                    }
                    black_box(&digests);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("hash_many", len), &len, |b, _| {
            b.iter(|| {
                Blake2s256.hash_many(black_box(&messages), black_box(&mut digests));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, blake2s_hash_many);
criterion_main!(benches);
