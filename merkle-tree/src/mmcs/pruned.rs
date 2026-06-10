//! The pruned (deduplicated) batch-opening data type.

use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// A batch opening with pruned Merkle authentication paths.
///
/// The pruned equivalent of opening multiple indices individually.
/// Shared sibling digests between queries are removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize, [D; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(
    deserialize = "T: serde::de::DeserializeOwned, [D; DIGEST_ELEMS]: serde::de::DeserializeOwned"
))]
pub struct PrunedBatchOpening<T, D, const DIGEST_ELEMS: usize> {
    /// Opened matrix rows for each query, in original input order.
    /// Outer index = query, middle index = matrix, inner = row elements.
    pub opened_values: Vec<Vec<Vec<T>>>,

    /// Compact authentication paths with redundant siblings removed.
    pub pruned_proof: crate::pruning::PrunedMerklePaths<D, DIGEST_ELEMS>,
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::{BatchOpeningRef, Mmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::{MerkleTreeError, MerkleTreeMmcs, PrunedProofError};

    type F = BabyBear;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;

    // 4-ary Poseidon2-based MMCS:
    //   - width-16 permutation for leaves
    //   - width-32 (4-to-1) permutation for internal compression
    type PermWide = Poseidon2BabyBear<32>;
    type MyCompress4 = TruncatedPermutation<PermWide, 4, 8, 32>;
    type MyMmcs4 =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress4, 4, 8>;

    // ----------------------------------------------------------------
    // Pruned opening integration tests
    // ----------------------------------------------------------------

    fn make_binary_mmcs(seed: u64) -> MyMmcs {
        let mut rng = SmallRng::seed_from_u64(seed);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        MyMmcs::new(hash, compress, 0)
    }

    fn make_4ary_mmcs_pruning(seed: u64) -> MyMmcs4 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let compress = MyCompress4::new(perm32);
        MyMmcs4::new(hash, compress, 0)
    }

    #[test]
    fn pruned_opening_matches_individual_binary() {
        let seed = 42u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 5, 7, 12, 15, 20, 31];

        // Open individually.
        let individual_openings: Vec<_> = indices
            .iter()
            .map(|&i| mmcs.open_batch(i, &prover_data))
            .collect();

        // Open pruned.
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        // Check opened values match.
        for (i, individual) in individual_openings.iter().enumerate() {
            assert_eq!(
                pruned_opening.opened_values[i], individual.opened_values,
                "opened values mismatch at query {i}"
            );
        }

        // Verify pruned opening (consumed by value — zero deep-clone).
        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_ok(), "pruned verification should succeed");
    }

    #[test]
    fn pruned_opening_4ary_roundtrip() {
        let seed = 99u64;
        let mmcs = make_4ary_mmcs_pruning(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 1, 10, 20, 30, 40, 50, 63];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_ok(), "4-ary pruned verification should succeed");
    }

    #[test]
    fn pruned_opening_rejects_tampered_proof() {
        let seed = 77u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 3, 7, 15];
        let mut pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        // Tamper with a sibling digest in the first path.
        if let Some(first_path) = pruned_opening.pruned_proof.paths.first_mut()
            && let Some(sibling) = first_path.siblings.first_mut()
        {
            sibling[0] = F::from_u32(999999);
        }

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_err(), "tampered proof should fail verification");
    }

    #[test]
    fn pruned_opening_rejects_more_paths_than_tree_height() {
        // A valid proof holds at most `max_height` unique paths:
        //   leaf indices are strictly ascending and distinct in `[0, max_height)`.
        // Claiming more is rejected before any per-path scratch is allocated.
        let seed = 123u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        // Height 8 -> max_height = 8.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 1, 2];
        let mut pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        // Inflate the unique-path count past the tree height by repeating a path.
        let filler = pruned_opening.pruned_proof.paths[0].clone();
        while pruned_opening.pruned_proof.paths.len() <= 8 {
            pruned_opening.pruned_proof.paths.push(filler.clone());
        }

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(
            matches!(
                result,
                Err(MerkleTreeError::MalformedPrunedProof(
                    PrunedProofError::TooManyUniquePaths {
                        got: 9,
                        max_height: 8
                    }
                ))
            ),
            "a proof with more unique paths than leaves should be rejected, got {result:?}"
        );
    }

    #[test]
    fn pruned_proof_is_smaller_than_individual() {
        let seed = 55u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 256, 4);
        let (_, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 1, 2, 3, 10, 11, 100, 101, 200, 201, 254, 255];

        // Count total siblings in individual proofs.
        let individual_total: usize = indices
            .iter()
            .map(|&i| mmcs.open_batch(i, &prover_data).opening_proof.len())
            .sum();

        // Count total siblings in pruned proof.
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);
        let pruned_total: usize = pruned_opening
            .pruned_proof
            .paths
            .iter()
            .map(|p| p.siblings.len())
            .sum();

        assert!(
            pruned_total < individual_total,
            "pruned {pruned_total} should be < individual {individual_total}"
        );
    }

    #[test]
    fn pruned_amortized_matches_per_path_h9() {
        // Sanity check on a non-power-of-two height: the amortized verifier
        // must accept exactly the same opening that the per-path verifier
        // accepts.
        let seed = 19363878127097954u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 9, 1);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![8, 7];

        for &i in &indices {
            let ind = mmcs.open_batch(i, &prover_data);
            let bref = BatchOpeningRef::new(&ind.opened_values, &ind.opening_proof);
            mmcs.verify_batch(&commit, &dims, i, bref).unwrap();
        }

        let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
        mmcs.verify_batch_pruned(&commit, &dims, pruned).unwrap();
    }

    #[test]
    fn pruned_amortized_clustered_queries_share_top_compressions() {
        // Queries clustered into two contiguous blocks share every layer
        // above their respective subtree roots — the amortized verifier
        // must collapse those shared compressions without changing the
        // accept/reject outcome.
        let seed = 11u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 1, 2, 3, 32, 33, 34, 35];
        let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
        mmcs.verify_batch_pruned(&commit, &dims, pruned).unwrap();
    }

    #[test]
    fn pruned_amortized_full_coverage_collapses_to_root() {
        // Opening every leaf in a small tree leaves exactly one group at the
        // top level: a single compression must reconstruct the cap.
        let seed = 13u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = (0..8).collect();
        let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
        mmcs.verify_batch_pruned(&commit, &dims, pruned).unwrap();
    }

    #[test]
    fn pruned_amortized_rejects_swapped_duplicate_opening() {
        // Pruned proofs collapse duplicate queries into one path, then the
        // verifier re-fans the opened values back out. If a malicious prover
        // swaps the opened values for a duplicate slot, the verifier must
        // reject — otherwise dedup would erase the discrepancy.
        let seed = 17u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 2);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![3, 7, 3];
        let mut pruned = mmcs.open_batch_pruned(&indices, &prover_data);
        // Swap one duplicate's opened values for a different leaf's values.
        let different = mmcs.open_batch(11, &prover_data);
        pruned.opened_values[2] = different.opened_values;

        let err = mmcs
            .verify_batch_pruned(&commit, &dims, pruned)
            .expect_err("mismatched duplicate openings must be rejected");
        // The duplicate is leaf index 3, which sorts to unique-path slot 0.
        assert!(matches!(
            err,
            MerkleTreeError::MalformedPrunedProof(
                PrunedProofError::InconsistentDuplicateOpenings { slot: 0 }
            )
        ));
    }

    #[test]
    fn pruned_amortized_cap_height_nonzero_binary() {
        // Invariant: surviving group digests land in cap[idx >> 4], not the root.
        //
        // Fixture state: binary tree, height 64, cap_height = 2.
        //
        //     schedule levels = 6 - 2 = 4
        //     cap entries     = 2^2   = 4
        let seed = 100u64;
        let mut rng = SmallRng::seed_from_u64(seed);

        // Hash + 2-to-1 compression for a binary tree.
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // 4-entry cap → schedule truncated by 2 layers.
        let mmcs = MyMmcs::new(hash, compress, 2);

        // Single matrix → fully binary schedule, no injections.
        let mut rng_mat = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng_mat, 64, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        // Hit every cap slot:
        //
        //     3, 7   → cap[0]    (0..16)
        //     23, 31 → cap[1]    (16..32)
        //     47     → cap[2]    (32..48)
        //     55, 63 → cap[3]    (48..64)
        let indices: Vec<usize> = vec![3, 7, 15, 23, 31, 47, 55, 63];

        let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
        mmcs.verify_batch_pruned(&commit, &dims, pruned).unwrap();
    }

    #[test]
    fn pruned_amortized_cap_height_with_mixed_heights() {
        // Invariant: cap truncation + matrix injection cooperate.
        //
        // Fixture state: binary, cap_height = 1, two matrices.
        //
        //     mat1: 32 rows × 4 cols   (injected at the leaf layer)
        //     mat2:  8 rows × 6 cols   (injected when padded height == 8)
        //     schedule levels = 5 - 1 = 4
        //     cap entries     = 2^1   = 2
        let seed = 200u64;
        let mut rng = SmallRng::seed_from_u64(seed);

        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // 2-entry cap → schedule truncated by 1 layer.
        let mmcs = MyMmcs::new(hash, compress, 1);

        // Height ratio 32 / 8 = 4 → 2 binary folds separate the injections.
        let mut rng_mat = SmallRng::seed_from_u64(seed);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng_mat, 32, 4);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng_mat, 8, 6);
        let dims = vec![mat1.dimensions(), mat2.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat1, mat2]);

        // Span the half-tree boundary (15 vs 16) to hit both cap entries.
        let indices: Vec<usize> = vec![0, 7, 15, 31];

        let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
        mmcs.verify_batch_pruned(&commit, &dims, pruned).unwrap();
    }

    #[test]
    fn pruned_amortized_4ary_cap_height_nonzero() {
        // Invariant: same as the binary cap test, but under 4-ary arity.
        //
        // Fixture state: 4-ary tree, height 64, cap_height = 1.
        //
        //     schedule levels = 3 - 1 = 2   (schedule = [4, 4])
        //     cap entries     = 4
        let seed = 300u64;
        let mut rng = SmallRng::seed_from_u64(seed);

        // Width-16 leaf hash + width-32 4-to-1 compression.
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let compress = MyCompress4::new(perm32);

        // 4-entry cap one layer below the root.
        let mmcs = MyMmcs4::new(hash, compress, 1);

        // Single 64-row matrix → pure 4-ary schedule, no injection.
        let mut rng_mat = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng_mat, 64, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        // Hit every cap slot:
        //
        //      1, 5  → cap[0]    ( 0..16)
        //     17     → cap[1]    (16..32)
        //     33     → cap[2]    (32..48)
        //     50, 63 → cap[3]    (48..64)
        let indices: Vec<usize> = vec![1, 5, 17, 33, 50, 63];

        let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
        mmcs.verify_batch_pruned(&commit, &dims, pruned).unwrap();
    }

    #[test]
    fn pruned_amortized_exhaustive_cap_height_sweep() {
        // Invariant: amortized verifier ≡ per-path verifier across every
        // (height, cap_height, query subset) cell in the sweep.
        //
        // Sweep:
        //
        //     height     ∈ {4, 8, 16, 32}
        //     cap_height ∈ 0..=log2(height)
        //     trials     = 32 per (height, cap_height)
        //
        // Failures carry (height, cap_height, trial) for localization.

        // Single RNG → seed alone reproduces any failure.
        let seed = 7777u64;
        let mut rng = SmallRng::seed_from_u64(seed);

        // Heights of 4 / 8 / 16 / 32 → schedules of 2 / 3 / 4 / 5 levels.
        for height in [4usize, 8, 16, 32] {
            // Walk cap heights from 0 (single root) up to log2(height) (empty schedule).
            for cap_height in 0..=p3_util::log2_ceil_usize(height) {
                // Fresh permutation per cell → no state leaks across cases.
                let perm = Perm::new_from_rng_128(&mut rng);
                let hash = MyHash::new(perm.clone());
                let compress = MyCompress::new(perm);
                let mmcs = MyMmcs::new(hash, compress, cap_height);

                // Pure binary schedule; only the cap shape varies per cell.
                let mat = RowMajorMatrix::<F>::rand(&mut rng, height, 3);
                let dims = vec![mat.dimensions()];
                let (commit, prover_data) = mmcs.commit(vec![mat]);

                // 32 trials → mix of scattered, clustered, dense, sparse subsets.
                for trial in 0..32 {
                    // num_queries ∈ [1, height-1] (never height, so the
                    // schedule still has work to do).
                    let num_queries = (trial % (height - 1)) + 1;

                    // Deterministic pseudo-random indices:
                    //
                    //     trial t, slot i  →  (i*13 + t*7) mod height
                    let indices: Vec<usize> = (0..num_queries)
                        .map(|i| (i * 13 + trial * 7) % height)
                        .collect();

                    // Oracle: per-path verifier on each index.
                    // Failure here = broken test setup, not amortized code.
                    for &i in &indices {
                        let ind = mmcs.open_batch(i, &prover_data);
                        let bref = BatchOpeningRef::new(&ind.opened_values, &ind.opening_proof);
                        mmcs.verify_batch(&commit, &dims, i, bref)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "per-path oracle failed: \
                                 height={height} cap={cap_height} trial={trial} index={i}: {e:?}"
                                )
                            });
                    }

                    // Subject: amortized verifier on the same set.
                    // Must accept (oracle just did).
                    let pruned = mmcs.open_batch_pruned(&indices, &prover_data);
                    mmcs.verify_batch_pruned(&commit, &dims, pruned)
                        .unwrap_or_else(|e| {
                            panic!(
                                "amortized pruned verifier failed: \
                             height={height} cap={cap_height} trial={trial}: {e:?}"
                            )
                        });
                }
            }
        }
    }

    #[test]
    fn pruned_opening_with_duplicate_indices() {
        let seed = 33u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        // Duplicate index 5.
        let indices: Vec<usize> = vec![5, 10, 5];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        // Should have 3 entries in opened_values (preserving duplicates).
        assert_eq!(pruned_opening.opened_values.len(), 3);
        assert_eq!(
            pruned_opening.opened_values[0], pruned_opening.opened_values[2],
            "duplicate queries should have identical values"
        );

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_ok());
    }

    #[test]
    fn pruned_opening_mixed_heights() {
        let seed = 44u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 8, 6);
        let dims = vec![mat1.dimensions(), mat2.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat1, mat2]);

        let indices: Vec<usize> = vec![0, 7, 15, 31];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(
            result.is_ok(),
            "mixed-height pruned verification should succeed"
        );
    }

    #[test]
    fn pruned_opening_4ary_mixed_heights() {
        // 4-ary MMCS with matrices at heights that don't align to the same
        // power of 4 — the schedule mixes 4-ary steps with binary bridges.
        let seed = 88u64;
        let mmcs = make_4ary_mmcs_pruning(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng, 64, 4);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 8, 6);
        let dims = vec![mat1.dimensions(), mat2.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat1, mat2]);

        let indices: Vec<usize> = vec![0, 5, 17, 33, 50, 63];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(
            result.is_ok(),
            "4-ary mixed-height pruned verification should succeed"
        );
    }

    // Pruned-opening proptests

    proptest! {
        #[test]
        fn proptest_pruned_binary_roundtrip(
            height in 2..128usize,
            width in 1..16usize,
            seed in 0u64..=u64::MAX,
            num_queries in 1..20usize,
        ) {
            // Invariant: pruned opening must produce the same opened values
            // and pass verification identically to individual openings.
            //
            // Fixture state: random binary tree (height x width) committed
            // with a seeded Poseidon2 permutation.

            // Build a binary MMCS and commit a random matrix.
            let mmcs = make_binary_mmcs(seed);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
            let dims = vec![mat.dimensions()];
            let (commit, prover_data) = mmcs.commit(vec![mat]);

            // Deterministic pseudo-random query indices (mod height).
            // Uses a large prime stride (7919) to spread queries across the tree.
            let indices: Vec<usize> = (0..num_queries)
                .map(|i| (seed as usize).wrapping_add(i * 7919) % height)
                .collect();

            // Reference: open each index individually (standard unpruned path).
            let individual: Vec<_> = indices
                .iter()
                .map(|&i| mmcs.open_batch(i, &prover_data))
                .collect();

            // Test subject: open all indices at once via pruning.
            let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

            // Check: opened values must be identical for every query.
            for (i, ind) in individual.iter().enumerate() {
                prop_assert_eq!(
                    &pruned_opening.opened_values[i],
                    &ind.opened_values,
                );
            }

            // Check: the pruned proof must verify against the commitment.
            mmcs.verify_batch_pruned(&commit, &dims, pruned_opening)
                .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
        }

        #[test]
        fn proptest_pruned_4ary_roundtrip(
            height in 2..128usize,
            width in 1..16usize,
            seed in 0u64..=u64::MAX,
            num_queries in 1..20usize,
        ) {
            // Same invariant as the binary test, but for 4-ary compression.
            // The arity schedule may include both 4-ary and binary bridge
            // steps depending on the matrix height.

            let mmcs = make_4ary_mmcs_pruning(seed);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
            let dims = vec![mat.dimensions()];
            let (commit, prover_data) = mmcs.commit(vec![mat]);

            let indices: Vec<usize> = (0..num_queries)
                .map(|i| (seed as usize).wrapping_add(i * 7919) % height)
                .collect();

            // Reference: individual openings.
            let individual: Vec<_> = indices
                .iter()
                .map(|&i| mmcs.open_batch(i, &prover_data))
                .collect();

            // Test subject: pruned opening.
            let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

            // Check: identical opened values.
            for (i, ind) in individual.iter().enumerate() {
                prop_assert_eq!(
                    &pruned_opening.opened_values[i],
                    &ind.opened_values,
                );
            }

            // Check: pruned proof verifies.
            mmcs.verify_batch_pruned(&commit, &dims, pruned_opening)
                .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
        }

        #[test]
        fn proptest_pruned_proof_size_leq_individual(
            height in 2..256usize,
            seed in 0u64..=u64::MAX,
            num_queries in 2..30usize,
        ) {
            // Invariant: the pruned proof must never contain more sibling
            // digests than the sum of all individual proofs.
            //
            // This should hold for any query pattern — clustered queries
            // save more, spread queries save less, but never go negative.

            let mmcs = make_binary_mmcs(seed);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mat = RowMajorMatrix::<F>::rand(&mut rng, height, 4);
            let (_, prover_data) = mmcs.commit(vec![mat]);

            let indices: Vec<usize> = (0..num_queries)
                .map(|i| (seed as usize).wrapping_add(i * 7919) % height)
                .collect();

            // Sum of individual proof sizes (unpruned baseline).
            let individual_total: usize = indices
                .iter()
                .map(|&i| mmcs.open_batch(i, &prover_data).opening_proof.len())
                .sum();

            // Pruned proof total.
            let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);
            let pruned_total: usize = pruned_opening
                .pruned_proof
                .paths
                .iter()
                .map(|p| p.siblings.len())
                .sum();

            prop_assert!(
                pruned_total <= individual_total,
                "pruned {} > individual {}", pruned_total, individual_total
            );
        }
    }
}
