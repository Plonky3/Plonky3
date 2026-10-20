//! Zerocheck prover and verifier on a chain of 32-bit adders filling `2^14` bits per instance.
//!
//! `FLOCK_TRACE=1` prints one proof's span tree before the measurements.
//!
//! On Linux the bench allocates through jemalloc with huge pages and retained extents, as the examples do.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, Poly64};
use p3_challenger::HashChallenger;
use p3_flock::zerocheck::{prove, verify};
use p3_flock::{Batch, Builder, Circuit, Wire, Witness};
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc's configuration, read before the first allocation.
#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: &[u8] = b"dirty_decay_ms:-1,thp:always,metadata_thp:always\0";

/// Fiat-Shamir sponge.
type Challenger = BinaryChallenger<Poly64, HashChallenger<u8, Keccak256Hash, 32>>;

/// Input words of one instance: eight 64-bit ports.
const INPUT_WORDS: usize = 8;

/// Products that keep the circuit just inside `2^14` bits.
const MAX_ANDS: usize = 15_700;

/// `x + y` over 32 bits by a ripple carry: one product per carry.
fn add32(c: &mut Builder, x: &[Wire], y: &[Wire]) -> Vec<Wire> {
    let mut carry = Wire::ZERO;
    let mut sum = Vec::with_capacity(32);
    for i in 0..32 {
        let t = c.xor(x[i], y[i]);
        sum.push(c.xor(t, carry));
        if i < 31 {
            // carry' = carry + (x + carry)(y + carry), the majority.
            let (xc, yc) = (c.xor(x[i], carry), c.xor(y[i], carry));
            let both = c.and(xc, yc);
            carry = c.xor(carry, both);
        }
    }
    sum
}

/// Sixteen 32-bit words mixed by additions, rotations and XORs until the products fill `2^14` bits.
fn adder_chain() -> Circuit {
    let mut c = Builder::new(&[64; INPUT_WORDS], &[64; 2]);
    let mut words: Vec<Vec<Wire>> = (0..INPUT_WORDS)
        .flat_map(|p| {
            let bits = c.input(p);
            [bits[..32].to_vec(), bits[32..].to_vec()]
        })
        .collect();
    let mut ands = 0;
    let mut i = 0;
    while ands + 31 <= MAX_ANDS {
        let n = words.len();
        let (x, y) = (&words[i % n], &words[(i + 5) % n]);
        // Rotate the second operand so the chain mixes bit positions.
        let rotated: Vec<Wire> = (0..32).map(|b| y[(b + 7 + i) % 32]).collect();
        let s = add32(&mut c, x, &rotated);
        words[i % n] = s;
        ands += 31;
        i += 1;
    }
    for (port, word) in words.iter().take(4).enumerate() {
        for (bit, &w) in word.iter().enumerate() {
            c.output(port / 2, 32 * (port % 2) + bit, w);
        }
    }
    c.finish()
}

/// The witness of `2^log_instances` random instances.
fn witness(circuit: &Circuit, log_instances: usize) -> Witness {
    let mut rng = SmallRng::seed_from_u64(7);
    let rows: Vec<[u64; INPUT_WORDS]> = (0..1usize << log_instances)
        .map(|_| core::array::from_fn(|_| rng.random()))
        .collect();
    Witness::generate(
        circuit,
        log_instances,
        &rows,
        &[0; INPUT_WORDS],
        |row, words| {
            words.copy_from_slice(row);
        },
    )
}

/// A fresh sponge.
const fn fresh() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

fn bench_zerocheck(c: &mut Criterion) {
    let circuit = adder_chain();
    assert_eq!(circuit.k_log(), 14, "the chain fills 2^14 bits");
    if std::env::var_os("FLOCK_TRACE").is_some() {
        Registry::default()
            .with(EnvFilter::new("info"))
            .with(ForestLayer::default())
            .init();
        let w = witness(&circuit, 18);
        let batch = Batch::new(&circuit, 18).expect("a zerocheck batch");
        let _ = prove(&batch, &w, 0, &mut fresh());
    }

    let mut group = c.benchmark_group("flock_zerocheck");
    group.sample_size(10);
    for log_instances in [16, 17, 18] {
        let w = witness(&circuit, log_instances);
        let batch = Batch::new(&circuit, log_instances).expect("a zerocheck batch");
        group.bench_function(BenchmarkId::new("prove", log_instances), |b| {
            b.iter(|| prove(&batch, &w, 0, &mut fresh()));
        });
        let (proof, _) = prove(&batch, &w, 0, &mut fresh());
        group.bench_function(BenchmarkId::new("verify", log_instances), |b| {
            b.iter(|| verify(&batch, &proof, 0, &mut fresh()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_zerocheck);
criterion_main!(benches);
