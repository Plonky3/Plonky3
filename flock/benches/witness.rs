//! Batch witness generation for each hash gadget, by the gate walk and by the gadget's generator.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_flock::gadgets::blake2s::Blake2sCompress;
use p3_flock::gadgets::blake3::Blake3Compress;
use p3_flock::gadgets::keccak::KeccakF1600;
use p3_flock::gadgets::sha256::Sha256Compress;
use p3_flock::{Gadget, Witness};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc's configuration, as the prover runs: freed pages stay mapped for the next batch.
#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: &[u8] = b"dirty_decay_ms:-1,thp:always,metadata_thp:always\0";

/// Instances per batch, as powers of two.
const LOG_NS: [usize; 2] = [16, 18];

/// Time one batch witness of a gadget, by the walk and by the gadget.
fn bench_gadget<G: Gadget + Sync>(
    c: &mut Criterion,
    name: &str,
    random: impl Fn(&mut SmallRng) -> G,
) {
    let circuit = G::circuit();
    let mut group = c.benchmark_group("witness");
    group.sample_size(10);
    for log_n in LOG_NS {
        let mut rng = SmallRng::seed_from_u64(0xF10C);
        let rows: Vec<G> = (0..1 << log_n).map(|_| random(&mut rng)).collect();
        group.throughput(Throughput::Elements(1 << log_n));
        group.bench_function(BenchmarkId::new(format!("{name}/walk"), log_n), |b| {
            b.iter(|| Witness::generate(&circuit, log_n, &rows, &rows[0], G::write_input));
        });
        group.bench_function(BenchmarkId::new(format!("{name}/gadget"), log_n), |b| {
            b.iter(|| G::witness(&circuit, log_n, &rows, &rows[0]));
        });
    }
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
