//! Figures for SLH-DSA-SHA2-128s signatures aggregated in one binary-field Flock proof.
//!
//! Every signature is of a 32-byte message under an empty context, by its own key.
//!
//! The opening targets 100 bits under list decoding up to the Johnson bound, with its own out-of-domain samples.
//!
//! Run with `RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --features parallel --example aggregate [k...]`.

use std::time::{Duration, Instant};

use p3_flock::{FlockChallenger, FlockConfig, PcsParams};
use p3_keccak::Keccak256Hash;
use p3_sphincs::{Aggregator, Claim, PUBLIC_KEY_BYTES, PublicKey, Signature, Statement};
use slh_dsa::{Sha2_128s, SigningKey};

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc's configuration, read before the first allocation: transparent huge pages, and freed pages returned at once.
#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: &[u8] =
    b"dirty_decay_ms:0,muzzy_decay_ms:0,thp:always,metadata_thp:always\0";

/// Timed runs per batch size, of which the median is printed.
const RUNS: usize = 3;

fn challenger() -> FlockChallenger {
    FlockChallenger::from_hasher(b"sphincs-aggregate".to_vec(), Keccak256Hash)
}

fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|k| k.parse().expect("a batch size"))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![1, 4, 16, 64, 128]
    } else {
        sizes
    };
    let config = FlockConfig {
        pcs: PcsParams {
            security_bits: 100,
            commitment_ood: true,
            ..PcsParams::default()
        },
        pow_bits: 0,
    };
    let start = Instant::now();
    let aggregator = Aggregator::new(config);
    let circuit = aggregator.circuit();
    println!(
        "SLH-DSA-SHA2-128s: {} SHA-256 compressions, {} products per signature, 2^{} bits, built in {:.2?}, {} threads",
        aggregator.compressions(),
        circuit.num_ands(),
        circuit.k_log(),
        start.elapsed(),
        std::thread::available_parallelism().map_or(1, usize::from)
    );

    let max = sizes.iter().copied().max().unwrap();
    let claims: Vec<Claim> = (0..max)
        .map(|i| {
            let tag = i as u8;
            let sk = SigningKey::<Sha2_128s>::slh_keygen_internal(
                &[tag; 16],
                &[!tag; 16],
                &[tag ^ 0x5A; 16],
            );
            let key: [u8; PUBLIC_KEY_BYTES] = sk.as_ref().to_bytes().as_slice().try_into().unwrap();
            let message = [tag; 32];
            let signature = sk
                .try_sign_with_context(&message, &[], None)
                .unwrap()
                .to_bytes();
            Claim {
                statement: Statement {
                    public_key: PublicKey::from_bytes(&key),
                    message,
                },
                signature: Signature::try_from(signature.as_slice()).unwrap(),
            }
        })
        .collect();
    let claim = &claims[0];
    let start = Instant::now();
    claim
        .statement
        .public_key
        .verify(&claim.statement.message, &[], &claim.signature)
        .unwrap();
    println!("native verify {:.2?}", start.elapsed());

    for k in sizes {
        let statements: Vec<Statement> = claims[..k].iter().map(|claim| claim.statement).collect();
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
            "k = {k:>3}: prove {proving:>9.2?} ({:.2?} per signature), verify {verifying:.2?}, proof {} KiB",
            proving / k as u32,
            size / 1024
        );
    }
}
