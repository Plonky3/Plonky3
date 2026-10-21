//! Batch witness generation for each hash gadget, 2^16 instances.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_flock::gadgets::blake2s::Blake2sCompress;
use p3_flock::gadgets::blake3::Blake3Compress;
use p3_flock::gadgets::keccak::KeccakF1600;
use p3_flock::gadgets::sha256::Sha256Compress;
use p3_flock::{Gadget, Witness};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

/// Instances per batch, as a power of two.
const LOG_N: usize = 16;

/// Time one batch witness of a gadget.
fn bench_gadget<G: Gadget + Sync>(
    c: &mut Criterion,
    name: &str,
    random: impl Fn(&mut SmallRng) -> G,
) {
    let mut rng = SmallRng::seed_from_u64(0xF10C);
    let rows: Vec<G> = (0..1 << LOG_N).map(|_| random(&mut rng)).collect();
    let circuit = G::circuit();
    let mut group = c.benchmark_group("witness");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1 << LOG_N));
    group.bench_function(BenchmarkId::new(name, LOG_N), |b| {
        b.iter(|| Witness::generate(&circuit, LOG_N, &rows, &rows[0], G::write_input));
    });
    group.finish();
}

/// Every hash gadget.
fn witness(c: &mut Criterion) {
    bench_gadget(c, "blake3", |rng| Blake3Compress {
        cv: rng.random(),
        block: rng.random(),
        counter: rng.next_u64(),
        block_len: 64,
        flags: 11,
    });
    bench_gadget(c, "blake2s", |rng| Blake2sCompress {
        h: rng.random(),
        m: rng.random(),
        t: rng.next_u64(),
        f0: u32::MAX,
    });
    bench_gadget(c, "sha256", |rng| Sha256Compress {
        state: rng.random(),
        block: rng.random(),
    });
    bench_gadget(c, "keccak_f1600", |rng| {
        KeccakF1600(core::array::from_fn(|_| rng.next_u64()))
    });
}

criterion_group!(benches, witness);
criterion_main!(benches);
