//! Figures for a binary recursion tree of keyed-BLAKE3 XMSS aggregation proofs at real parameters.
//!
//! Keys have a lifetime of 2^20 epochs, `w = 4` and 64 chains; every signer signs one message at one epoch.
//!
//! Four leaf proofs of `n` signatures each, two level-1 node proofs, one root: depth two, `4 n` signatures.
//!
//! A level-1 instance must fit 2^27 bits, which allows up to 128 signatures per leaf.
//!
//! Run with `cargo run --release --features parallel --example aggregate_tree -- n`.

use std::sync::Arc;
use std::time::Instant;

use p3_blake3::Blake3;
use p3_flock::recursion::{
    Child, Matrix, NodeLayout, NodeProof, build_node, leaf_digest, verify_root,
};
use p3_flock::{Batch, FlockChallenger, FlockConfig, FlockProof, PublicPorts};
use p3_maybe_rayon::prelude::*;
use p3_xmss::{BinaryAggregator, Blake3Hash, Claim, Params, Statement, Xmss};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc's configuration: freed pages go back at once, so the peak is the live set.
#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: &[u8] = b"dirty_decay_ms:0,muzzy_decay_ms:0\0";

/// Leaf proofs of the tree.
const LEAVES: usize = 4;

/// The epoch every signer signs at.
const EPOCH: u32 = 3;

/// Peak resident memory so far, in GB.
fn peak_gb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb: f64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    kb / (1024.0 * 1024.0)
}

/// Serialized size in KiB.
fn kib(proof: &FlockProof<Blake3>) -> usize {
    postcard::to_allocvec(proof).expect("serialize").len() / 1024
}

/// Print a node's figures.
fn report(level: usize, node: &NodeProof, seconds: f64) {
    println!(
        "level {level}: instance 2^{} bits, {} ANDs, {} ops, {} terms, built and proved in {seconds:.1} s, proof {} KiB, peak {:.1} GB",
        node.circuit.circuit.k_log(),
        node.circuit.ands,
        node.circuit.ops,
        node.circuit.terms,
        kib(&node.proof),
        peak_gb()
    );
    for cost in &node.circuit.costs {
        println!("  {:<17} {:>11} ANDs", cost.component, cost.ands);
    }
}

/// `n` signatures by fresh signers on one message, proved in one leaf over a BLAKE3 transcript.
fn leaf(
    aggregator: &BinaryAggregator,
    xmss: &Xmss<Blake3Hash>,
    seed: u64,
    n: usize,
) -> (Vec<Statement<Blake3Hash>>, FlockProof<Blake3>, f64) {
    let message = [seed as u8; 32];
    let claims: Vec<Claim<Blake3Hash>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed << 32 | i as u64);
            let (public_key, secret) = xmss.generate(&mut rng, EPOCH as usize + 1);
            let signature = xmss
                .sign(&secret, EPOCH, &message)
                .expect("an active epoch");
            Claim {
                public_key,
                epoch: EPOCH,
                message,
                signature,
            }
        })
        .collect();
    let start = Instant::now();
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let proof = aggregator
        .prove(&claims, &mut challenger)
        .expect("valid signatures");
    let seconds = start.elapsed().as_secs_f64();
    (
        claims.iter().map(Claim::statement).collect(),
        proof,
        seconds,
    )
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(128, |a| a.parse().expect("signatures per leaf"));
    let params = Params::with_lifetime(20);
    let xmss = Xmss::with_hash(params, Blake3Hash);
    // Rate 1/16 cuts the queries a verifier circuit replays.
    let mut config = FlockConfig::default();
    config.pcs.log_inv_rate = 4;
    // Leaves trade prover work for verifier work: rate 1/256 cuts the queries the level-1 circuit replays by a fifth.
    let mut leaf_config = config;
    leaf_config.pcs.log_inv_rate = 8;
    let aggregator = BinaryAggregator::new(params, leaf_config);
    let circuit = aggregator.circuit();
    println!(
        "XMSS 2^20 epochs, w = 4, 64 chains: {} ANDs per signature, 2^{} bits; {n} signatures per leaf, {} threads",
        circuit.num_ands(),
        circuit.k_log(),
        current_num_threads()
    );
    let layout = NodeLayout {
        slots: 3,
        k_max: 30,
        advice_words: Some(24_000),
        k_log: Some(27),
    };
    let total = Instant::now();

    let leaves: Vec<_> = (0..LEAVES as u64)
        .map(|seed| {
            let (statements, proof, seconds) = leaf(&aggregator, &xmss, seed, n);
            println!(
                "leaf {seed}: proved in {seconds:.1} s, proof {} KiB, peak {:.1} GB",
                kib(&proof),
                peak_gb()
            );
            (statements, proof)
        })
        .collect();
    let statements: Vec<(Batch<'_>, PublicPorts)> = leaves
        .iter()
        .map(|(s, _)| aggregator.statement(s).expect("a batch"))
        .collect();
    let children: Vec<Child<'_>> = statements
        .iter()
        .zip(&leaves)
        .map(|((batch, public), (_, proof))| Child::Leaf {
            config: &leaf_config,
            batch: *batch,
            public,
            proof,
        })
        .collect();
    let leaf_matrix = Arc::new(Matrix::new(circuit));

    let mut level1: Vec<NodeProof> = Vec::new();
    for pair in children.chunks(2) {
        let start = Instant::now();
        // Both level-1 nodes have one circuit: the second reuses the first's, compiled once.
        let reuse = level1.first().map(|n| &n.circuit.circuit);
        let node = build_node(
            &layout,
            [&pair[0], &pair[1]],
            &[Arc::clone(&leaf_matrix)],
            reuse,
        )
        .and_then(|node| node.prove(&config))
        .expect("a level-1 node");
        report(1, &node, start.elapsed().as_secs_f64());
        level1.push(node);
    }
    drop(children);

    let level1_matrix = Arc::new(Matrix::new(&level1[0].circuit.circuit));
    let matrices = [Arc::clone(&leaf_matrix), level1_matrix];
    let start = Instant::now();
    let root = build_node(
        &layout,
        [
            &Child::Node {
                config: &config,
                node: &level1[0],
            },
            &Child::Node {
                config: &config,
                node: &level1[1],
            },
        ],
        &matrices,
        None,
    )
    .and_then(|node| node.prove(&config))
    .expect("the root");
    report(2, &root, start.elapsed().as_secs_f64());
    let tree_seconds = total.elapsed().as_secs_f64();
    drop(level1);

    let publics: Vec<&PublicPorts> = statements.iter().map(|(_, p)| p).collect();
    let start = Instant::now();
    verify_root(&layout, &config, &root, &matrices, &leaf_digest(&publics)).expect("the tree");
    let check = start.elapsed().as_secs_f64();
    let signatures = LEAVES * n;
    println!(
        "{signatures} signatures: tree built and proved in {tree_seconds:.1} s ({:.1} ms per signature), final check {check:.1} s, root proof {} KiB, peak {:.1} GB",
        1e3 * tree_seconds / signatures as f64,
        kib(&root.proof),
        peak_gb()
    );
}
