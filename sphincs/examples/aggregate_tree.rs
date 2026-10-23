//! Figures for a binary recursion tree whose leaves are SLH-DSA-SHA2-128s aggregation proofs over a BLAKE3 transcript.
//!
//! Each leaf proves `n` signatures; two leaves feed one level-1 node, two level-1 nodes one root.
//!
//! Run with `cargo run --release --features parallel --example aggregate_tree -- n`.

use std::sync::Arc;
use std::time::Instant;

use p3_blake3::Blake3;
use p3_flock::recursion::{
    Child, Matrix, NodeLayout, NodeProof, build_node, leaf_digest, verify_root,
};
use p3_flock::{Batch, FlockChallenger, FlockConfig, FlockProof, PublicPorts};
use p3_sphincs::{Aggregator, Claim, PUBLIC_KEY_BYTES, PublicKey, Signature, Statement};
use slh_dsa::{Sha2_128s, SigningKey};

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
}

/// A claim of a deterministic key over `[tag; 32]`.
fn claim(tag: u8) -> Claim {
    let sk =
        SigningKey::<Sha2_128s>::slh_keygen_internal(&[tag; 16], &[!tag; 16], &[tag ^ 0x5A; 16]);
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
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(1, |a| a.parse().expect("signatures per leaf"));
    let mut config = FlockConfig::default();
    config.pcs.log_inv_rate = 4;
    let start = Instant::now();
    let aggregator = Aggregator::new(config);
    let circuit = aggregator.circuit();
    println!(
        "SLH-DSA-SHA2-128s: {} SHA-256 compressions, {} ANDs, {} ops, {} terms per signature, 2^{} bits, built in {:.1} s, peak {:.1} GB",
        aggregator.compressions(),
        circuit.num_ands(),
        circuit.num_ops(),
        circuit.num_terms(),
        circuit.k_log(),
        start.elapsed().as_secs_f64(),
        peak_gb()
    );
    let layout = NodeLayout {
        slots: 3,
        k_max: 30,
        advice_words: Some(24_000),
        k_log: Some(27),
    };
    let total = Instant::now();
    let leaves: Vec<(Vec<Statement>, FlockProof<Blake3>)> = (0..LEAVES)
        .map(|leaf| {
            let claims: Vec<Claim> = (0..n).map(|i| claim((leaf * n + i) as u8)).collect();
            let start = Instant::now();
            let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
            let proof = aggregator
                .prove(&claims, &mut challenger)
                .expect("valid signatures");
            println!(
                "leaf {leaf}: proved in {:.1} s, proof {} KiB, peak {:.1} GB",
                start.elapsed().as_secs_f64(),
                kib(&proof),
                peak_gb()
            );
            (claims.iter().map(|c| c.statement).collect(), proof)
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
            config: &config,
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
    let signatures = LEAVES * n;
    println!(
        "{signatures} signatures: tree built and proved in {tree_seconds:.1} s ({:.1} s per signature), final check {:.1} s, root proof {} KiB, peak {:.1} GB",
        tree_seconds / signatures as f64,
        start.elapsed().as_secs_f64(),
        kib(&root.proof),
        peak_gb()
    );
}
