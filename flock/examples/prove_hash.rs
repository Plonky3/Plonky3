//! Prove a batch of hash compressions end to end and print per-phase times.
//!
//! With `--public` every input and output word is public, so the proof attests to these hashes.
//!
//! Without it the proof binds no input or output: a throughput benchmark that attests to no particular hash.
//!
//! Usage: `prove_hash <blake3|blake2s|sha256|keccak> <log_instances> [runs] [--public]`.

use std::time::{Duration, Instant};

use p3_flock::gadgets::blake2s::Blake2sCompress;
use p3_flock::gadgets::blake3::Blake3Compress;
use p3_flock::gadgets::keccak::KeccakF1600;
use p3_flock::gadgets::sha256::Sha256Compress;
use p3_flock::{
    Batch, FlockChallenger, FlockConfig, Gadget, Port, PublicPorts, Witness, prove, verify,
};
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
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

/// A fresh sponge.
fn sponge() -> FlockChallenger {
    FlockChallenger::from_hasher(b"flock-bench".to_vec(), Keccak256Hash)
}

/// Median of a run.
fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

/// Time witness, prove and verify of one gadget over `runs` runs.
fn run<G: Gadget + Sync>(
    name: &str,
    log_n: usize,
    runs: usize,
    public: bool,
    random: impl Fn(&mut SmallRng) -> G,
) {
    let mut rng = SmallRng::seed_from_u64(0xF10C);
    let rows: Vec<G> = (0..1usize << log_n).map(|_| random(&mut rng)).collect();
    let circuit = G::circuit();
    let batch = Batch::new(&circuit, log_n).expect("a batch the zerocheck takes");
    let config = FlockConfig::default();
    let (mut wit, mut pro, mut ver) = (vec![], vec![], vec![]);
    let mut size = 0;
    let mut statement = PublicPorts::default();
    let mut build = Duration::ZERO;
    for _ in 0..runs {
        let t = Instant::now();
        let witness = Witness::generate(&circuit, log_n, &rows, &rows[0], G::write_input);
        wit.push(t.elapsed());

        // Every input and output word, the outputs read from the witness.
        if public && statement.is_empty() {
            let words = (1 << circuit.k_log()) / 64;
            let ports: Vec<Port> = (0..)
                .map(Port::Input)
                .take_while(|&p| circuit.port(p).is_some())
                .chain(
                    (0..)
                        .map(Port::Output)
                        .take_while(|&p| circuit.port(p).is_some()),
                )
                .collect();
            let t = Instant::now();
            statement = PublicPorts::new(&batch, &ports, |i, out| {
                let len = out.len();
                out.copy_from_slice(&witness.z()[i * words..][..len]);
            });
            build = t.elapsed();
        }

        let t = Instant::now();
        let proof = prove(&config, &batch, &statement, witness, &mut sponge()).expect("prove");
        pro.push(t.elapsed());

        let t = Instant::now();
        verify(&config, &batch, &statement, &proof, &mut sponge()).expect("verify");
        ver.push(t.elapsed());
        size = postcard_len(&proof);
    }
    let (w, p, v) = (median(wit), median(pro), median(ver));
    let n = (1usize << log_n) as f64;
    println!(
        "{name}{} 2^{log_n} (k_log {}, {} ANDs): witness {w:?}, prove {p:?}, total {:?}, {:.0} k/s, verify {v:?}, proof {} KiB",
        if public {
            format!(" public (statement {build:?})")
        } else {
            String::new()
        },
        circuit.k_log(),
        circuit.num_ands(),
        w + p,
        n / (w + p).as_secs_f64() / 1000.0,
        size / 1024,
    );
}

/// Serialized proof size in bytes.
fn postcard_len(proof: &p3_flock::FlockProof) -> usize {
    postcard::to_allocvec(proof).expect("serialize").len()
}

fn main() {
    // Spans print only when RUST_LOG asks for them.
    Registry::default()
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::OFF.into())
                .from_env_lossy(),
        )
        .with(ForestLayer::default())
        .init();
    let args: Vec<String> = std::env::args().collect();
    let hash = args.get(1).map_or("blake3", String::as_str);
    let log_n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let public = args.iter().any(|a| a == "--public");
    match hash {
        "blake3" => run("blake3", log_n, runs, public, |rng| Blake3Compress {
            cv: rng.random(),
            block: rng.random(),
            counter: rng.next_u64(),
            block_len: 64,
            flags: 11,
        }),
        "blake2s" => run("blake2s", log_n, runs, public, |rng| Blake2sCompress {
            h: rng.random(),
            m: rng.random(),
            t: rng.next_u64(),
            f0: u32::MAX,
        }),
        "sha256" => run("sha256", log_n, runs, public, |rng| Sha256Compress {
            state: rng.random(),
            block: rng.random(),
        }),
        "keccak" => run("keccak", log_n, runs, public, |rng| {
            KeccakF1600(core::array::from_fn(|_| rng.next_u64()))
        }),
        other => panic!("unknown hash {other}"),
    }
}
