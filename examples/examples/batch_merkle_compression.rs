//! Demonstrates batch Merkle proof compression.
//!
//! This example shows how opening multiple indices from a Merkle tree can be
//! compressed by deduplicating shared sibling nodes (Octopus algorithm).
//!
//! Run with: cargo run --example batch_merkle_compression

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{BatchMerkleProof, MerkleTreeMmcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::SeedableRng;

type F = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 8>;

fn main() {
    println!("=== Batch Merkle Proof Compression Demo ===\n");

    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm);
    let mmcs = ValMmcs::new(hash, compress);

    // Test different configurations
    for log_height in [10, 14, 18] {
        for num_queries in [50, 100, 200] {
            test_compression(&mmcs, log_height, num_queries, &mut rng);
        }
    }
}

fn test_compression(mmcs: &ValMmcs, log_height: usize, num_queries: usize, rng: &mut SmallRng) {
    let height = 1 << log_height;
    let width = 8;

    // Create random matrix
    let mat = RowMajorMatrix::<F>::rand(rng, height, width);
    let (_, prover_data) = mmcs.commit(vec![mat]);

    // Generate random query indices (simulating FRI queries)
    let indices: Vec<usize> = (0..num_queries)
        .map(|i| {
            // Pseudo-random but deterministic indices
            ((i * 7919 + 104729) % height) as usize
        })
        .collect();

    // Collect individual proofs
    let individual_openings: Vec<_> = indices
        .iter()
        .map(|&idx| mmcs.open_batch(idx, &prover_data))
        .collect();

    let individual_proofs: Vec<Vec<[F; 8]>> = individual_openings
        .iter()
        .map(|opening| opening.opening_proof.clone())
        .collect();

    // Calculate individual proof size
    let individual_digests: usize = individual_proofs.iter().map(|p| p.len()).sum();
    let individual_bytes = individual_digests * 8 * 4; // 8 field elements * 4 bytes each

    // Create batch proof
    let batch_proof = BatchMerkleProof::from_single_proofs(&individual_proofs, &indices);
    let batch_digests = batch_proof.num_digests();
    let batch_bytes = batch_digests * 8 * 4;

    // Calculate savings
    let savings_pct = 100.0 * (1.0 - batch_digests as f64 / individual_digests as f64);
    let bytes_saved = individual_bytes - batch_bytes;

    println!(
        "Height: 2^{:<2} | Queries: {:>3} | Individual: {:>6} digests ({:>7} bytes) | Batch: {:>5} digests ({:>7} bytes) | Saved: {:>5.1}% ({:>6} bytes)",
        log_height,
        num_queries,
        individual_digests,
        individual_bytes,
        batch_digests,
        batch_bytes,
        savings_pct,
        bytes_saved
    );
}
