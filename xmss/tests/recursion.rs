//! Two keyed-BLAKE3 XMSS aggregation proofs aggregated into one node proof of a binary recursion tree.

use std::sync::Arc;
use std::time::Instant;

use p3_blake3::Blake3;
use p3_flock::recursion::{
    Child, Matrix, NodeLayout, NodeProof, build_node, leaf_digest, verify_root,
};
use p3_flock::{FlockChallenger, FlockConfig, FlockProof, PublicPorts};
use p3_xmss::{BinaryAggregator, Blake3Hash, Claim, Params, Statement, Xmss};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Signatures per leaf proof.
const SIGNATURES: usize = 2;

/// A leaf: the aggregation proof of `SIGNATURES` fresh signatures, over a BLAKE3 transcript and Merkle tree.
fn leaf(
    aggregator: &BinaryAggregator,
    xmss: &Xmss<Blake3Hash>,
    seed: u64,
) -> (Vec<Statement<Blake3Hash>>, FlockProof<Blake3>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let claims: Vec<Claim<Blake3Hash>> = (0..SIGNATURES)
        .map(|i| {
            let (public_key, secret) = xmss.generate(&mut rng, 4);
            let (epoch, message) = (i as u32, [seed as u8 ^ i as u8; 32]);
            let signature = xmss.sign(&secret, epoch, &message).unwrap();
            Claim {
                public_key,
                epoch,
                message,
                signature,
            }
        })
        .collect();
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let start = Instant::now();
    let proof = aggregator.prove(&claims, &mut challenger).unwrap();
    let proving = start.elapsed();
    let statements: Vec<Statement<Blake3Hash>> = claims.iter().map(Claim::statement).collect();
    let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
    let start = Instant::now();
    aggregator
        .verify(&statements, &proof, &mut challenger)
        .unwrap();
    println!(
        "leaf: {SIGNATURES} signatures proved in {proving:?}, verified natively in {:?}, proof {} KiB",
        start.elapsed(),
        postcard::to_allocvec(&proof).unwrap().len() / 1024
    );
    (statements, proof)
}

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

/// Print a node's figures.
fn report(level: usize, node: &NodeProof, seconds: f64) {
    println!(
        "level {level}: instance 2^{} bits, {} ANDs, {} gates, built and proved in {seconds:.1} s, proof {} KiB, peak {:.1} GB",
        node.circuit.circuit.k_log(),
        node.circuit.ands,
        node.circuit.gates,
        postcard::to_allocvec(&node.proof).unwrap().len() / 1024,
        peak_gb()
    );
    for cost in &node.circuit.costs {
        println!(
            "  {:<17} {:>11} ANDs {:>12} gates",
            cost.component, cost.ands, cost.gates
        );
    }
}

#[test]
#[ignore = "proves a node circuit of 2^26 bits per instance (peak under 20 GB); run with --release --include-ignored"]
fn two_xmss_aggregation_proofs_aggregate_into_one() {
    // A lifetime of four epochs and sixteen chains keep the leaf proof, and so its verifier circuit, small.
    let params = Params::new(2, 2, 16, 24);
    let xmss = Xmss::with_hash(params, Blake3Hash);
    // Rate 1/16 cuts the queries a verifier circuit replays.
    let mut config = FlockConfig::default();
    config.pcs.log_inv_rate = 4;
    let aggregator = BinaryAggregator::new(params, config);
    let layout = NodeLayout {
        slots: 3,
        k_max: 30,
        advice_words: None,
        k_log: None,
    };

    let leaves: Vec<_> = (1..=2).map(|seed| leaf(&aggregator, &xmss, seed)).collect();
    let statements: Vec<(p3_flock::Batch<'_>, PublicPorts)> = leaves
        .iter()
        .map(|(statements, _)| aggregator.statement(statements).unwrap())
        .collect();
    let leaf_matrix = Arc::new(Matrix::new(aggregator.circuit()));
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

    let start = Instant::now();
    let level1 = build_node(
        &layout,
        [&children[0], &children[1]],
        &[Arc::clone(&leaf_matrix)],
        None,
    )
    .unwrap()
    .prove(&config)
    .unwrap();
    report(1, &level1, start.elapsed().as_secs_f64());
    let expected = leaf_digest(&[&statements[0].1, &statements[1].1]);
    let start = Instant::now();
    verify_root(
        &layout,
        &config,
        &level1,
        &[Arc::clone(&leaf_matrix)],
        &expected,
    )
    .unwrap();
    println!(
        "level 1 final check {:?}, peak {:.1} GB",
        start.elapsed(),
        peak_gb()
    );

    // A wrong leaf statement is refused by the final check.
    let mut other = leaves[1].0.clone();
    other[0].message[0] ^= 1;
    let (_, other_public) = aggregator.statement(&other).unwrap();
    let wrong = leaf_digest(&[&statements[0].1, &other_public]);
    assert!(
        verify_root(
            &layout,
            &config,
            &level1,
            &[Arc::clone(&leaf_matrix)],
            &wrong
        )
        .is_err()
    );
}
