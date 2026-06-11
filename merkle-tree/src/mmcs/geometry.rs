//! Tree geometry: height-ladder validation and width checks.

use alloc::vec::Vec;

use p3_matrix::Dimensions;
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};

use crate::MerkleTreeError;
use crate::MerkleTreeError::{EmptyBatch, IncompatibleHeights, WrongWidth};

/// Check that each opened row has exactly the width of its matrix.
///
/// The leaf hash flattens all rows at one height into a single element stream,
/// so a digest match alone does not pin where one row ends and the next begins.
pub(crate) fn check_widths<T>(
    dimensions: &[Dimensions],
    opened_values: &[Vec<T>],
) -> Result<(), MerkleTreeError> {
    for (matrix, (dims, row)) in dimensions.iter().zip(opened_values).enumerate() {
        if row.len() != dims.width {
            return Err(WrongWidth {
                matrix,
                expected: dims.width,
                got: row.len(),
            });
        }
    }
    Ok(())
}

/// The arity schedule and query positions for a given Merkle path.
/// This is used to replay the arity schedule for a given Merkle path and recover, for each
/// tree level, both:
/// - the compression arity `step` used at that level (either `N` or `2`), and
/// - the position of the queried child within its group, `pos_in_group`, in
///   the range `0..step`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArityAndPositions {
    /// The arity schedule for the Merkle path.
    pub arity_schedule: Vec<usize>,
    /// The query positions for the Merkle path.
    pub query_positions: Vec<usize>,
}

/// Validates that a batch of matrix heights forms a tree geometry the commitment can build.
///
/// # Overview
///
/// - The tallest matrix fixes the leaf layer of the tree.
/// - At `k` halvings above the leaves, the tree covers `ceil(max_height / 2^k)` rows.
/// - A shorter matrix can only inject at depth `k` if its height is exactly that value.
///
/// ```text
///     max_height = 7:
///       k = 0 → 7 rows   (leaf layer)
///       k = 1 → 4 rows   (ceil(7 / 2))
///       k = 2 → 2 rows   (ceil(7 / 4))
///       k = 3 → 1 row    (ceil(7 / 8))
///
///     valid heights: {7, 4, 2, 1} — anything else has no injection layer.
/// ```
///
/// Both sides of the protocol apply this rule:
/// - committing refuses to build a tree from incompatible heights,
/// - verifying refuses dimension claims that no commitment could have produced.
///
/// # Returns
///
/// The tallest height in the batch.
///
/// # Errors
///
/// - A batch with no matrices, or only zero-height matrices, has no tree.
/// - Any height off the reachable ladder is rejected,
///   with the offending and expected heights attached.
///
/// # Performance
///
/// - Two passes over the heights, no allocation.
/// - Runs on every batch verification, so it must stay cheap.
pub(crate) fn validate_commit_reachable_heights<I>(heights: I) -> Result<usize, MerkleTreeError>
where
    I: IntoIterator<Item = usize>,
    I::IntoIter: Clone,
{
    let heights = heights.into_iter();

    // Pass 1: find the tallest matrix; it anchors the leaf layer.
    // A missing maximum means no matrices at all.
    // A zero maximum means every matrix is empty — equally unusable.
    let max_height = heights
        .clone()
        .max()
        .filter(|&height| height > 0)
        .ok_or(EmptyBatch)?;
    let log_max_height = log2_ceil_usize(max_height);

    // Pass 2: pin every height to the one rung of the ladder it claims.
    for height in heights {
        // Number of halvings between the leaf layer and this matrix's injection layer.
        // Heights in the same power-of-two bucket share the same depth,
        // so at most one height per bucket can be valid.
        //
        // A zero height maps to depth `log_max_height`,
        // where the expected value below is 1, so it is rejected.
        let bits_reduced = log_max_height - log2_ceil_usize(height);

        // `((m - 1) >> k) + 1` computes `ceil(m / 2^k)` without shift overflow.
        let expected_height = ((max_height - 1) >> bits_reduced) + 1;

        if height != expected_height {
            return Err(IncompatibleHeights {
                height,
                max_height,
                expected_height,
                bits_reduced,
            });
        }
    }

    Ok(max_height)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::{BatchOpeningRef, Mmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::{MerkleTreeError, MerkleTreeMmcs};

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

    #[test]
    #[should_panic]
    fn mismatched_heights() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // attempt to commit to a mat with 8 rows and a mat with 7 rows. this should panic.
        let large_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7, 8].map(F::from_u8).to_vec(), 1);
        let small_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7].map(F::from_u8).to_vec(), 1);
        let _ = mmcs.commit(vec![large_mat, small_mat]);
    }

    #[test]
    #[should_panic(expected = "matrix height 3 incompatible with tallest height 7")]
    fn commit_rejects_unreachable_height() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // Invariant: every shorter matrix must sit on the ladder ceil(7 / 2^k).
        //
        //     k = 0 → 7    k = 1 → 4    k = 2 → 2    k = 3 → 1
        //
        // Height 3 is on no rung, so no global leaf index can map into it.
        let tall = RowMajorMatrix::<F>::rand(&mut rng, 7, 1);
        let unreachable = RowMajorMatrix::<F>::rand(&mut rng, 3, 1);

        // Committing is prover-side: an impossible shape is a caller bug → panic.
        mmcs.commit(vec![tall, unreachable]);
    }

    #[test]
    fn verifier_rejects_dimensions_that_commit_cannot_produce() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // Fixture state: a valid commitment to heights [7, 4].
        // 4 = ceil(7 / 2), so the short matrix injects one halving above the leaves.
        let tall = RowMajorMatrix::<F>::rand(&mut rng, 7, 1);
        let short = RowMajorMatrix::<F>::rand(&mut rng, 4, 1);
        let (commit, prover_data) = mmcs.commit(vec![tall, short]);

        // The opening itself is honest; only the claimed dimensions will lie.
        let opening = mmcs.open_batch(6, &prover_data);

        // Mutation: claim the short matrix has height 3 instead of 4.
        //
        //     committed heights: [7, 4]   ladder of 7: {7, 4, 2, 1}
        //     claimed heights:   [7, 3]   3 is on no rung
        //
        // Heights 3 and 4 pad to the same power of two, so the path replay
        // walks identical layers; only the geometry rule can tell them apart.
        let impossible_dims = vec![
            Dimensions {
                height: 7,
                width: 1,
            },
            Dimensions {
                height: 3,
                width: 1,
            },
        ];

        let err = mmcs
            .verify_batch(&commit, &impossible_dims, 6, (&opening).into())
            .expect_err("verifier must reject dimensions that the commit path cannot produce");

        // The rejection names the offending height and the only valid height
        // at its depth: one halving below 7 covers ceil(7 / 2) = 4 rows, never 3.
        match err {
            MerkleTreeError::IncompatibleHeights {
                height,
                max_height,
                expected_height,
                bits_reduced,
            } => {
                assert_eq!(height, 3);
                assert_eq!(max_height, 7);
                assert_eq!(expected_height, 4);
                assert_eq!(bits_reduced, 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn height_validation_edge_cases() {
        // No matrices → no tree to build.
        assert!(matches!(
            validate_commit_reachable_heights(core::iter::empty()),
            Err(MerkleTreeError::EmptyBatch)
        ));

        // Zero-height matrices contribute no rows;
        // an all-zero batch is as empty as no batch at all.
        assert!(matches!(
            validate_commit_reachable_heights([0, 0]),
            Err(MerkleTreeError::EmptyBatch)
        ));

        // A zero height next to real rows is unreachable:
        // every rung of the ladder ceil(8 / 2^k) is at least 1.
        //
        //     ladder of 8: {8, 4, 2, 1}   claimed: 0 → depth k = 3 expects 1
        assert!(matches!(
            validate_commit_reachable_heights([8, 0]),
            Err(MerkleTreeError::IncompatibleHeights {
                height: 0,
                max_height: 8,
                expected_height: 1,
                bits_reduced: 3,
            })
        ));

        // A single matrix sits at depth 0 by definition; its height is returned.
        assert_eq!(validate_commit_reachable_heights([5]).unwrap(), 5);

        // A full ladder validates: every rung ceil(7 / 2^k) = {7, 4, 2, 1}.
        assert_eq!(validate_commit_reachable_heights([7, 4, 2, 1]).unwrap(), 7);

        // Duplicates on a rung are fine: both matrices inject at the same layer.
        assert_eq!(validate_commit_reachable_heights([7, 4, 4]).unwrap(), 7);
    }

    #[test]
    #[should_panic(expected = "matrix height 5 incompatible")]
    fn commit_rejects_missing_leaf_coverage() {
        let mut rng = SmallRng::seed_from_u64(9);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        let tallest = RowMajorMatrix::new(vec![F::ONE; 11], 1);
        let invalid = RowMajorMatrix::new(vec![F::ONE; 5], 1);

        // We expect a panic because the smaller matrix needs ceil(11 / 2) == 6 rows;
        // height 5 would leave the global index 5 unmapped at that layer.
        let _ = mmcs.commit(vec![tallest, invalid]);
    }

    #[test]
    fn replay_arity_and_positions_binary() {
        let mut rng = SmallRng::seed_from_u64(123);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // Use a moderately sized tree to ensure multiple levels.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = vec![mat.dimensions()];
        let (_commit, prover_data) = mmcs.commit(vec![mat]);

        // For a range of indices, the helper's schedule should match the tree's
        // actual arity_schedule and the naive index-based positions.
        for index in [0usize, 1, 5, 16, 31] {
            let opening = mmcs.open_batch(index, &prover_data);
            let (_opened_values, proof) = opening.unpack();

            let arity_and_positions = mmcs
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            let steps = arity_and_positions.arity_schedule;
            let positions = arity_and_positions.query_positions;

            // With cap_height = 0, we expect one step per non-root layer.
            let expected_levels = prover_data.digest_layers.len().saturating_sub(1);
            assert_eq!(steps.len(), expected_levels);
            assert_eq!(positions.len(), expected_levels);

            // Steps must match the concrete arity_schedule stored in the tree.
            for (i, &step) in steps.iter().enumerate() {
                assert_eq!(
                    step, prover_data.arity_schedule[i],
                    "step mismatch at level {i} for index {index}"
                );
            }

            // Positions must agree with repeatedly dividing the index by the step.
            let mut idx = index;
            for (level, (&step, &pos_in_group)) in steps.iter().zip(&positions).enumerate() {
                let expected_pos = idx % step;
                assert_eq!(
                    pos_in_group, expected_pos,
                    "pos_in_group mismatch at level {level} for index {index}"
                );
                idx /= step;
            }
        }
    }

    #[test]
    fn replay_arity_and_positions_4ary() {
        let mut rng = SmallRng::seed_from_u64(456);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        // Height chosen so that both N-ary and possible binary bridge steps can appear.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![mat.dimensions()];
        let (_commit, prover_data) = mmcs4.commit(vec![mat]);

        for index in [0usize, 3, 7, 17, 42, 63] {
            let opening = mmcs4.open_batch(index, &prover_data);
            let (_opened_values, proof) = opening.unpack();

            let arity_and_positions = mmcs4
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            let steps = arity_and_positions.arity_schedule;
            let positions = arity_and_positions.query_positions;

            let expected_levels = prover_data.digest_layers.len().saturating_sub(1);
            assert_eq!(steps.len(), expected_levels);
            assert_eq!(positions.len(), expected_levels);

            // Each step must be either 2 (binary) or 4 (full 4-ary) and match the
            // concrete arity_schedule stored in the Merkle tree.
            for (i, &step) in steps.iter().enumerate() {
                assert!(
                    step == 2 || step == 4,
                    "unexpected step {step} at level {i} for index {index}"
                );
                assert_eq!(
                    step, prover_data.arity_schedule[i],
                    "step mismatch at level {i} for index {index}"
                );
            }

            // Positions must be consistent with index reduction at each level.
            let mut idx = index;
            for (level, (&step, &pos_in_group)) in steps.iter().zip(&positions).enumerate() {
                let expected_pos = idx % step;
                assert_eq!(
                    pos_in_group, expected_pos,
                    "pos_in_group mismatch at level {level} for index {index}"
                );
                idx /= step;
            }
        }
    }

    #[test]
    fn replay_arity_and_positions_4ary_mixed_heights() {
        let mut rng = SmallRng::seed_from_u64(789);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        let mat64 = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let mat16 = RowMajorMatrix::<F>::rand(&mut rng, 16, 6);
        let mat4 = RowMajorMatrix::<F>::rand(&mut rng, 4, 5);
        let dims = vec![mat64.dimensions(), mat16.dimensions(), mat4.dimensions()];
        let (_commit, prover_data) = mmcs4.commit(vec![mat64, mat16, mat4]);

        for index in [0usize, 3, 7, 17, 42, 63] {
            let opening = mmcs4.open_batch(index, &prover_data);
            let (_opened_values, proof) = opening.unpack();

            let arity_and_positions = mmcs4
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            let steps = arity_and_positions.arity_schedule;
            let positions = arity_and_positions.query_positions;

            let expected_levels = prover_data.digest_layers.len().saturating_sub(1);
            assert_eq!(steps.len(), expected_levels);
            assert_eq!(positions.len(), expected_levels);

            for (i, &step) in steps.iter().enumerate() {
                assert_eq!(
                    step, prover_data.arity_schedule[i],
                    "step mismatch at level {i} for index {index}"
                );
            }

            let mut idx = index;
            for (level, (&step, &pos_in_group)) in steps.iter().zip(&positions).enumerate() {
                let expected_pos = idx % step;
                assert_eq!(
                    pos_in_group, expected_pos,
                    "pos_in_group mismatch at level {level} for index {index}"
                );
                idx /= step;
            }
        }
    }

    #[test]
    fn replay_arity_and_positions_4ary_non_power_of_two_target() {
        let mut rng = SmallRng::seed_from_u64(790);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        // This shape forces an intermediate padded height of 12, where 12 / 4 = 3
        // is not a power of two. The prover and verifier must still pick the same
        // step schedule.
        let mat33 = RowMajorMatrix::<F>::rand(&mut rng, 33, 8);
        let mat3 = RowMajorMatrix::<F>::rand(&mut rng, 3, 5);
        let dims = vec![mat33.dimensions(), mat3.dimensions()];
        let (commit, prover_data) = mmcs4.commit(vec![mat33, mat3]);

        assert!(
            prover_data.arity_schedule.iter().all(|&step| step == 4),
            "expected full 4-ary schedule for this shape"
        );

        for index in [0usize, 1, 16, 32] {
            let opening = mmcs4.open_batch(index, &prover_data);
            let (opened_values, proof) = opening.unpack();

            mmcs4
                .verify_batch(
                    &commit,
                    &dims,
                    index,
                    BatchOpeningRef::new(&opened_values, &proof),
                )
                .expect("verification should succeed for non-power-of-two target case");

            let arity_and_positions = mmcs4
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            assert_eq!(
                arity_and_positions.arity_schedule, prover_data.arity_schedule,
                "replayed schedule must match prover schedule at index {index}"
            );
        }
    }

    mod proptests {
        use alloc::vec::Vec;

        use proptest::prelude::*;

        use super::*;

        type PermWide = Poseidon2BabyBear<32>;
        type MyCompress4 = TruncatedPermutation<PermWide, 4, 8, 32>;
        type MyMmcs4 =
            MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress4, 4, 8>;

        fn make_binary_mmcs(seed: u64) -> MyMmcs {
            let mut rng = SmallRng::seed_from_u64(seed);
            let perm = Perm::new_from_rng_128(&mut rng);
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm);
            MyMmcs::new(hash, compress, 0)
        }

        fn make_4ary_mmcs(seed: u64) -> MyMmcs4 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let perm16 = Perm::new_from_rng_128(&mut rng);
            let perm32 = PermWide::new_from_rng_128(&mut rng);
            let hash = MyHash::new(perm16);
            let compress = MyCompress4::new(perm32);
            MyMmcs4::new(hash, compress, 0)
        }

        fn matrix_strategy() -> impl Strategy<Value = (usize, usize, u64)> {
            (1..=256_usize, 1..=32_usize, 0u64..=u64::MAX)
        }

        proptest! {
            #[test]
            fn proptest_ladder_heights_always_accepted(
                max_height in 1usize..=1 << 12,
                k in 0usize..=12,
            ) {
                // Every rung ceil(max_height / 2^k) is a shape a commitment can build,
                // so validation must accept it alongside the tallest matrix.
                let height = max_height.div_ceil(1 << k);
                prop_assert!(
                    validate_commit_reachable_heights([max_height, height]).is_ok()
                );
            }

            #[test]
            fn proptest_height_validation_matches_ladder(
                max_height in 1usize..=1 << 12,
                height in 1usize..=1 << 12,
            ) {
                prop_assume!(height <= max_height);

                // Independent oracle: enumerate the ladder by direct ceiling division.
                //
                //     ladder = { ceil(max_height / 2^k) : k = 0..=12 }
                //
                // Validation uses the shift identity ((m - 1) >> k) + 1 and derives
                // the depth from the height's own power-of-two bucket; the two
                // formulations must agree on every input.
                let on_ladder = (0..=12).any(|k| max_height.div_ceil(1usize << k) == height);
                let accepted =
                    validate_commit_reachable_heights([max_height, height]).is_ok();
                prop_assert_eq!(accepted, on_ladder);
            }

            #[test]
            fn proptest_binary_merkle_roundtrip((height, width, seed) in matrix_strategy()) {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
                let dims = vec![mat.dimensions()];
                let mmcs = make_binary_mmcs(seed.wrapping_add(1));

                let (commit, prover_data) = mmcs.commit(vec![mat.clone()]);

                let index = (seed as usize) % height;
                let opening = mmcs.open_batch(index, &prover_data);

                let (opened_values, proof) = opening.unpack();
                mmcs.verify_batch(&commit, &dims, index, BatchOpeningRef::new(&opened_values, &proof))
                    .expect("binary MerkleTreeMmcs verify should succeed");

                let expected_row: Vec<F> = mat.row(index).unwrap().into_iter().collect();
                prop_assert_eq!(&opened_values[0], &expected_row, "opened row should match");
            }

            #[test]
            fn proptest_4ary_merkle_roundtrip((height, width, seed) in matrix_strategy()) {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
                let dims = vec![mat.dimensions()];
                let mmcs = make_4ary_mmcs(seed.wrapping_add(1));

                let (commit, prover_data) = mmcs.commit(vec![mat.clone()]);

                let index = (seed as usize) % height;
                let opening = mmcs.open_batch(index, &prover_data);

                let (opened_values, proof) = opening.unpack();
                mmcs.verify_batch(&commit, &dims, index, BatchOpeningRef::new(&opened_values, &proof))
                    .expect("4-ary MerkleTreeMmcs verify should succeed");

                let expected_row: Vec<F> = mat.row(index).unwrap().into_iter().collect();
                prop_assert_eq!(&opened_values[0], &expected_row, "opened row should match");
            }

            #[test]
            fn proptest_4ary_all_indices((height, seed) in (1..=64_usize, 0u64..=u64::MAX)) {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mat = RowMajorMatrix::<F>::rand(&mut rng, height, 8);
                let dims = vec![mat.dimensions()];
                let mmcs = make_4ary_mmcs(seed.wrapping_add(1));
                let (commit, prover_data) = mmcs.commit(vec![mat.clone()]);

                for index in 0..height {
                    let opening = mmcs.open_batch(index, &prover_data);
                    let (opened_values, proof) = opening.unpack();
                    mmcs.verify_batch(&commit, &dims, index, BatchOpeningRef::new(&opened_values, &proof))
                        .expect("4-ary verify at each index should succeed");
                    let expected: Vec<F> = mat.row(index).unwrap().into_iter().collect();
                    prop_assert_eq!(&opened_values[0], &expected);
                }
            }
        }
    }
}
