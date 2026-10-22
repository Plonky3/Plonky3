//! Figures for keyed-BLAKE3 XMSS signatures aggregated in one binary-field Flock proof.
//!
//! Keys have a lifetime of 2^20 epochs, of which the first eight are active, and signatures carry the full 20-level path.
//!
//! The opening targets 100 bits under list decoding up to the Johnson bound, with its own out-of-domain samples.
//!
//! Run with `RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --features parallel --example aggregate_binary`.

use std::time::{Duration, Instant};

use p3_flock::{FlockChallenger, FlockConfig, PcsParams};
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::current_num_threads;
use p3_xmss::{BinaryAggregator, Blake3Hash, Claim, Params, Statement, Xmss};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc's configuration, read before the first allocation: transparent huge pages for the witness tables.
#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: &[u8] = b"dirty_decay_ms:-1,thp:always,metadata_thp:always\0";

/// Timed runs per batch size, of which the median is printed.
const RUNS: usize = 5;

fn challenger() -> FlockChallenger {
    FlockChallenger::from_hasher(b"xmss-aggregate-binary".to_vec(), Keccak256Hash)
}

fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

fn main() {
    let params = Params::with_lifetime(20);
    let xmss = Xmss::with_hash(params, Blake3Hash);
    let config = FlockConfig {
        pcs: PcsParams {
            security_bits: 100,
            commitment_ood: true,
            ..PcsParams::default()
        },
        pow_bits: 0,
    };
    let aggregator = BinaryAggregator::new(params, config);
    let circuit = aggregator.circuit();
    println!(
        "keyed BLAKE3, w = 4, 64 chains, target 96, 2^20 epochs: {} products per signature, 2^{} bits, {} threads",
        circuit.num_ands(),
        circuit.k_log(),
        current_num_threads()
    );

    let max = 1024;
    let mut rng = StdRng::seed_from_u64(0);
    let claims: Vec<Claim<Blake3Hash>> = (0..max)
        .map(|i| {
            let (public_key, secret) = xmss.generate(&mut rng, 8);
            let (epoch, message) = (5, [i as u8; 32]);
            let signature = xmss.sign(&secret, epoch, &message).unwrap();
            Claim {
                public_key,
                epoch,
                message,
                signature,
            }
        })
        .collect();
    let claim = &claims[0];
    let start = Instant::now();
    xmss.verify(
        &claim.public_key,
        claim.epoch,
        &claim.message,
        &claim.signature,
    )
    .unwrap();
    println!("native verify {:.2?}", start.elapsed());

    for k in [1, 16, 64, 256, 1024] {
        let statements: Vec<Statement<Blake3Hash>> =
            claims[..k].iter().map(Claim::statement).collect();
        let (mut proving, mut verifying) = (Vec::new(), Vec::new());
        let mut size = 0;
        for _ in 0..RUNS {
            let start = Instant::now();
            let proof = aggregator.prove(&claims[..k], &mut challenger()).unwrap();
            proving.push(start.elapsed());
            let start = Instant::now();
            aggregator
                .verify(&statements, &proof, &mut challenger())
                .unwrap();
            verifying.push(start.elapsed());
            size = postcard::to_allocvec(&proof).unwrap().len();
        }
        let (proving, verifying) = (median(proving), median(verifying));
        println!(
            "k = {k:>4}: prove {proving:>9.2?} ({:.2?} per signature), verify {verifying:.2?}, proof {} KiB",
            proving / k as u32,
            size / 1024
        );
    }
}
