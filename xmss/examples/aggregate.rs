//! Figures for XMSS signatures verified in leaf circuits and aggregated by a recursive node.
//!
//! Keys have a lifetime of 2^20 epochs, of which the first eight are active, so key generation stays cheap.
//!
//! Signatures still carry the full 20-level path, so every circuit has the shape of a 2^20 lifetime.
//!
//! Proofs use FRI with blowup 2, 100 queries and 16 bits of query grinding: 116 conjectured bits.
//!
//! Run with `--release --features parallel` for multithreaded proving.

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::{Duration, Instant};

use common::{Config, EF, F, Proven, config};
use p3_batch_stark::{ProverData, prove_batch, verify_batch};
use p3_circuit::recursion::aggregate;
use p3_circuit::tables::CircuitAir;
use p3_circuit::{Circuit, KeyBinding};
use p3_field::PrimeCharacteristicRing;
use p3_fri::FriParameters;
use p3_xmss::{Chip, Claim, Params, Xmss, chip, leaf_circuit};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// The mean time of `runs` calls.
fn mean<T>(runs: u32, mut f: impl FnMut() -> T) -> Duration {
    let start = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(f());
    }
    start.elapsed() / runs
}

/// The table name and padded height of every table.
fn rows(circuit: &Circuit<F, EF, Chip>) -> String {
    circuit
        .airs
        .iter()
        .zip(&circuit.traces)
        .map(|(air, trace)| {
            let name = match air {
                CircuitAir::Value(_) => "value",
                CircuitAir::Public(_) => "public",
                CircuitAir::BaseAlu(_) => "base-alu",
                CircuitAir::ExtAlu(_) => "ext-alu",
                CircuitAir::Recompose(_) => "recompose",
                CircuitAir::Permute(_) => "permute",
            };
            format!("{name} 2^{}", (trace.values.len() / trace.width).ilog2())
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Prove `circuit`, print its figures, and return its proof.
fn prove(config: &Config, label: &str, circuit: Circuit<F, EF, Chip>, signatures: usize) -> Proven {
    println!("{label}: {}", rows(&circuit));
    let instances = circuit.instances::<Config>();
    let start = Instant::now();
    let prover_data = ProverData::from_instances(config, &instances).unwrap();
    let setup = start.elapsed();
    let start = Instant::now();
    let proof = prove_batch(config, &instances, &prover_data).unwrap();
    let proving = start.elapsed();
    drop(instances);
    let public_values = circuit.public_values();
    let start = Instant::now();
    verify_batch(
        config,
        &circuit.airs,
        &proof,
        &public_values,
        &prover_data.common,
    )
    .unwrap();
    let verifying = start.elapsed();
    let size = postcard::to_allocvec(&proof).unwrap().len();
    println!(
        "  setup {setup:.2?}, prove {proving:.2?} ({:.2?} per signature), verify {verifying:.2?}, proof {} KiB",
        proving / signatures as u32,
        size / 1024
    );
    Proven {
        airs: circuit.airs,
        common: prover_data.common,
        proof,
        public_values,
    }
}

fn main() {
    let params = Params::with_lifetime(20);
    let xmss = Xmss::new(params);
    let config = config(FriParameters::new_benchmark);
    let mut rng = StdRng::seed_from_u64(0);
    let (epoch, message) = (5, core::array::from_fn(F::from_usize));

    // Every signer signs the same message at the same epoch.
    let start = Instant::now();
    let keys: Vec<_> = (0..32).map(|_| xmss.generate(&mut rng, 8)).collect();
    let keygen = start.elapsed() / 32;
    let claims: Vec<Claim> = keys
        .iter()
        .map(|(public_key, secret)| Claim {
            public_key: *public_key,
            epoch,
            message,
            signature: xmss.sign(secret, epoch, &message).unwrap(),
        })
        .collect();
    let (public, secret) = &keys[0];
    let signing = mean(100, || xmss.sign(secret, epoch, &message));
    let verifying = mean(100, || {
        xmss.verify(public, epoch, &message, &claims[0].signature)
    });
    println!(
        "native (w = 4, 64 chains, target 96, 2^20 epochs): keygen {keygen:.2?} for 8 active epochs, sign {signing:.2?}, verify {verifying:.2?}"
    );

    let mut leaves = Vec::new();
    for k in [1, 4, 16] {
        let start = Instant::now();
        let circuit = leaf_circuit::<EF, _>(chip(), &params, &claims[..k]).unwrap();
        println!("leaf, {k} signatures, built in {:.2?}", start.elapsed());
        leaves.push(prove(&config, "  tables", circuit, k));
    }

    // A node verifies two 16-signature leaves.
    let second = leaf_circuit::<EF, _>(chip(), &params, &claims[16..]).unwrap();
    let second = prove(&config, "second 16-signature leaf", second, 16);
    let children = [leaves[2].child(), second.child()];
    let start = Instant::now();
    let node = aggregate(chip(), &config, KeyBinding::Constant, &children).unwrap();
    println!(
        "node, 2 leaves of 16 signatures, built in {:.2?}",
        start.elapsed()
    );
    prove(&config, "  tables", node, 32);
}
