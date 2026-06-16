//! The standard [`Mmcs`] interface: commit, open, and verify a single index.

use alloc::vec::Vec;
use core::cmp::Reverse;

use itertools::Itertools;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};

use super::{check_widths, validate_commit_reachable_heights};
use crate::MerkleTreeError::{
    CapMismatch, EmptyBatch, IndexOutOfBounds, WrongBatchSize, WrongHeight,
};
use crate::merkle_tree::padded_len;
use crate::{MerkleCap, MerkleTree, MerkleTreeError, MerkleTreeMmcs};

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> Mmcs<P::Value>
    for MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    P: PackedValue,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    PW::Value: Eq + Clone,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = MerkleTree<P::Value, PW::Value, M, N, DIGEST_ELEMS>;
    type Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>;
    type Proof = Vec<[PW::Value; DIGEST_ELEMS]>;
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        // Geometry gate: refuse to build a tree from heights off the
        // `ceil(max_height / 2^k)` ladder.
        // Committing is prover-side, so a bad shape is a caller bug → panic.
        // The error's display text carries the offending and expected heights.
        match validate_commit_reachable_heights(inputs.iter().map(|m| m.height())) {
            Ok(_) => {}
            Err(EmptyBatch) => panic!("all matrices have height 0"),
            Err(err) => panic!("{err}"),
        }

        let tree = MerkleTree::new::<P, PW, H, C>(&self.hash, &self.compress, inputs);

        // Make cap_height fit this tree (small trees during FRI folding may have fewer layers)
        let num_layers = tree.num_layers();
        let effective_cap_height = self.cap_height.min(num_layers.saturating_sub(1));

        let cap = tree.cap(effective_cap_height);
        (cap, tree)
    }

    /// Opens a batch of rows from committed matrices.
    ///
    /// Returns `(openings, proof)` where `openings` is a vector whose `i`th element is
    /// the `j`th row of the ith matrix `M[i]`, and `proof` is the vector of sibling
    /// Merkle tree nodes allowing the verifier to reconstruct the committed cap entry.
    ///
    /// At each tree level the number of siblings is `step - 1`, where `step` is either
    /// `N` (full N-ary compression) or `2` (binary) at levels that sit between N-ary layers).
    /// For binary levels the remaining `N - 2` inputs are padded with the default digest.
    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &MerkleTree<P::Value, PW::Value, M, N, DIGEST_ELEMS>,
    ) -> BatchOpening<P::Value, Self> {
        let max_height = self.get_max_height(prover_data);
        assert!(
            index < max_height,
            "index {index} out of bounds for height {max_height}"
        );
        let log_max_height = log2_ceil_usize(max_height);

        // Get the matrix rows encountered along the path from the cap to the given leaf index.
        let openings = prover_data
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.row(reduced_index).unwrap().into_iter().collect()
            })
            .collect_vec();

        // Only include siblings up to (but not including) the cap layer.
        let num_layers = prover_data.digest_layers.len();
        let effective_cap_height = self.cap_height.min(num_layers.saturating_sub(1));
        let proof_levels = num_layers
            .saturating_sub(1)
            .saturating_sub(effective_cap_height);

        let mut proof = Vec::new();
        let mut idx = index;
        for layer_idx in 0..proof_levels {
            let step = prover_data.arity_schedule[layer_idx];
            let group_start = (idx / step) * step;
            let pos_in_group = idx % step;
            for k in 0..step {
                if k != pos_in_group {
                    proof.push(prover_data.digest_layers[layer_idx][group_start + k]);
                }
            }
            idx /= step;
        }

        BatchOpening::new(openings, proof)
    }

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.leaves.iter().collect()
    }

    /// Verifies an opened batch of rows with respect to a given commitment (Merkle cap).
    ///
    /// At each tree level, the verifier determines whether this level used a full
    /// N-ary step or a binary step (when a matrix injection sits between N-ary
    /// layers). Binary steps carry `1` sibling in the proof; N-ary steps carry
    /// `N-1` siblings. Both cases use the same N-to-1 compression function —
    /// binary steps pad the remaining `N-2` slots with the default digest.
    ///
    /// # Arguments
    /// - `commit`: The Merkle cap of the tree.
    /// - `dimensions`: A vector of the dimensions of the matrices committed to.
    ///   Each shorter matrix height must equal `ceil(max_height / 2^k)` for some `k`,
    ///   matching the power-of-two reduction that maps global leaf indices into that matrix.
    ///   Heights are only bound up to their power-of-two padding:
    ///   two valid shapes whose padded layers coincide verify against the same cap.
    ///   Callers must therefore source dimensions from public data, never from the prover.
    /// - `index`: The index of a leaf in the tree.
    /// - `batch_proof`: A reference to a batched opening proof, containing:
    ///   - `opened_values`: A vector of matrix rows. Assume that the tallest matrix committed
    ///     to has height `max_height`, then the `j`th opened value must be the row
    ///     `Mj[index >> k]`, where `k` is the power-of-two reduction depth for `Mj`.
    ///   - `opening_proof`: A vector of sibling nodes. The `i`th element should be the node at level `i`
    ///     with index `(index << i) ^ 1`.
    ///
    /// # Returns
    /// `Ok(())` if the verification is successful; otherwise returns a verification error.
    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        mut index: usize,
        batch_proof: BatchOpeningRef<'_, P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_proof.unpack();
        // Check that the openings have the correct shape.
        if dimensions.len() != opened_values.len() {
            return Err(WrongBatchSize);
        }
        // Geometry gate: the claimed heights must form a tree the commitment can build.
        // The tallest height anchors the index bound and the leaf layer width below.
        let max_height =
            validate_commit_reachable_heights(dimensions.iter().map(|dims| dims.height))?;

        // Derive the arity schedule from verifier-known dimensions and the configured cap height.
        // Rejecting a wrong-length proof here costs only integer work — no hashing yet.
        let arity_schedule = self.proof_arity_schedule(dimensions)?;
        let expected_proof_len: usize = arity_schedule.iter().map(|step| step - 1).sum();
        if opening_proof.len() != expected_proof_len {
            return Err(WrongHeight {
                expected_proof_len,
                num_siblings: opening_proof.len(),
            });
        }

        // The leaf hash flattens all rows at one height into a single stream,
        // so row boundaries are only authenticated by checking each row width.
        check_widths(dimensions, opened_values)?;

        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        // Leaf layer width before the walk starts, padded to a full N-ary group.
        let mut curr_height_padded = padded_len(max_height, N);

        if index >= max_height {
            return Err(IndexOutOfBounds { max_height, index });
        }

        // Hash all matrix openings at the current height.
        let leaf_height_npt = max_height.next_power_of_two();
        let mut digest: [PW::Value; DIGEST_ELEMS] = self.hash.hash_iter_slices(
            heights_tallest_first
                .peeking_take_while(|(_, dims)| dims.height.next_power_of_two() == leaf_height_npt)
                .map(|(i, _)| opened_values[i].as_slice()),
        );

        let default_digest = [PW::Value::default(); DIGEST_ELEMS];

        let mut proof_pos: usize = 0;

        for &step in &arity_schedule {
            let num_siblings = step - 1;
            let siblings = &opening_proof[proof_pos..proof_pos + num_siblings];
            proof_pos += num_siblings;

            let pos_in_group = index % step;
            let mut sibling_idx = 0;
            let inputs: [_; N] = core::array::from_fn(|k| {
                if k < step {
                    if k == pos_in_group {
                        digest
                    } else {
                        let s = siblings[sibling_idx];
                        sibling_idx += 1;
                        s
                    }
                } else {
                    default_digest
                }
            });

            digest = self.compress.compress(inputs);
            index /= step;
            let logical_next = curr_height_padded / step;
            curr_height_padded = padded_len(logical_next, N);

            // Check if there are any new matrix rows to inject at the next height.
            let logical_next_npt = logical_next.next_power_of_two();
            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == logical_next_npt);
            if let Some(next_height) = next_height {
                let next_height_openings_digest = self.hash.hash_iter_slices(
                    heights_tallest_first
                        .peeking_take_while(|(_, dims)| dims.height == next_height)
                        .map(|(i, _)| opened_values[i].as_slice()),
                );

                let inject_inputs: [_; N] = core::array::from_fn(|k| {
                    if k == 0 {
                        digest
                    } else if k == 1 {
                        next_height_openings_digest
                    } else {
                        default_digest
                    }
                });
                digest = self.compress.compress(inject_inputs);
            }
        }

        // After processing the proof, `index` has been shifted by the proof length.
        // This index now points into the cap layer.
        let cap_index = index;
        if cap_index < commit.num_roots() && commit[cap_index] == digest {
            Ok(())
        } else {
            Err(CapMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::{BatchOpeningRef, Mmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};
    use p3_symmetric::{
        CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
    };
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

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
    fn commit_single_1x8() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);

        // v = [2, 1, 2, 2, 0, 0, 1, 0]
        let v = vec![
            F::TWO,
            F::ONE,
            F::TWO,
            F::TWO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let (commit, _) = mmcs.commit_vec(v.clone());

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([hash.hash_item(v[0]), hash.hash_item(v[1])]),
                compress.compress([hash.hash_item(v[2]), hash.hash_item(v[3])]),
            ]),
            compress.compress([
                compress.compress([hash.hash_item(v[4]), hash.hash_item(v[5])]),
                compress.compress([hash.hash_item(v[6]), hash.hash_item(v[7])]),
            ]),
        ]);
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn poseidon2_4ary_single_matrix_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Leaf hasher: width-16
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);

        // Internal compression: width-32, 4-to-1.
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);

        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs4.commit(vec![mat]);

        let index = 17;
        let opening = mmcs4.open_batch(index, &prover_data);
        mmcs4
            .verify_batch(&commit, &dims, index, (&opening).into())
            .expect("4-ary Poseidon2 MMCS roundtrip should verify");
    }

    #[test]
    fn small_height_4ary_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(99);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        // max_height=2 with N=4: edge case where max_height < N.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 2, 8);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs4.commit(vec![mat]);

        for index in 0..2 {
            let opening = mmcs4.open_batch(index, &prover_data);
            mmcs4
                .verify_batch(&commit, &dims, index, (&opening).into())
                .expect("small-height 4-ary roundtrip should verify");
        }
    }

    #[test]
    fn commit_single_8x1() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress, 0);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 1, 8);
        let (commit, _) = mmcs.commit(vec![mat.clone()]);

        let expected_result = hash.hash_iter(mat.vertically_packed_row(0));
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn commit_single_2x2() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);

        // mat = [
        //   0 1
        //   2 1
        // ]
        let mat = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE], 2);

        let (commit, _) = mmcs.commit(vec![mat]);

        let expected_result = compress.compress([
            hash.hash_slice(&[F::ZERO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::ONE]),
        ]);
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn commit_single_2x3() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);
        let default_digest = [F::ZERO; 8];

        // mat = [
        //   0 1
        //   2 1
        //   2 2
        // ]
        let mat = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE, F::TWO, F::TWO], 2);

        let (commit, _) = mmcs.commit(vec![mat]);

        let expected_result = compress.compress([
            compress.compress([
                hash.hash_slice(&[F::ZERO, F::ONE]),
                hash.hash_slice(&[F::TWO, F::ONE]),
            ]),
            compress.compress([hash.hash_slice(&[F::TWO, F::TWO]), default_digest]),
        ]);
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn commit_mixed() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);
        let default_digest = [F::ZERO; 8];

        // mat_1 = [
        //   0 1
        //   2 1
        //   2 2
        //   2 1
        //   2 2
        // ]
        let mat_1 = RowMajorMatrix::new(
            vec![
                F::ZERO,
                F::ONE,
                F::TWO,
                F::ONE,
                F::TWO,
                F::TWO,
                F::TWO,
                F::ONE,
                F::TWO,
                F::TWO,
            ],
            2,
        );
        // mat_2 = [
        //   1 2 1
        //   0 2 2
        //   1 2 1
        // ]
        let mat_2 = RowMajorMatrix::new(
            vec![
                F::ONE,
                F::TWO,
                F::ONE,
                F::ZERO,
                F::TWO,
                F::TWO,
                F::ONE,
                F::TWO,
                F::ONE,
            ],
            3,
        );

        let (commit, prover_data) = mmcs.commit(vec![mat_1, mat_2]);

        let mat_1_leaf_hashes = [
            hash.hash_slice(&[F::ZERO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::TWO]),
            hash.hash_slice(&[F::TWO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::TWO]),
        ];
        let mat_2_leaf_hashes = [
            hash.hash_slice(&[F::ONE, F::TWO, F::ONE]),
            hash.hash_slice(&[F::ZERO, F::TWO, F::TWO]),
            hash.hash_slice(&[F::ONE, F::TWO, F::ONE]),
        ];

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([
                    compress.compress([mat_1_leaf_hashes[0], mat_1_leaf_hashes[1]]),
                    mat_2_leaf_hashes[0],
                ]),
                compress.compress([
                    compress.compress([mat_1_leaf_hashes[2], mat_1_leaf_hashes[3]]),
                    mat_2_leaf_hashes[1],
                ]),
            ]),
            compress.compress([
                compress.compress([
                    compress.compress([mat_1_leaf_hashes[4], default_digest]),
                    mat_2_leaf_hashes[2],
                ]),
                default_digest,
            ]),
        ]);

        assert_eq!(commit[0], expected_result);

        let (opened_values, _) = mmcs.open_batch(2, &prover_data).unpack();
        assert_eq!(
            opened_values,
            vec![vec![F::TWO, F::TWO], vec![F::ZERO, F::TWO, F::TWO]]
        );
    }

    #[test]
    fn commit_either_order() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        let input_1 = RowMajorMatrix::<F>::rand(&mut rng, 5, 8);
        let input_2 = RowMajorMatrix::<F>::rand(&mut rng, 3, 16);

        let (commit_1_2, _) = mmcs.commit(vec![input_1.clone(), input_2.clone()]);
        let (commit_2_1, _) = mmcs.commit(vec![input_2, input_1]);
        assert_eq!(commit_1_2, commit_2_1);
    }

    #[test]
    fn verify_tampered_proof_fails() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // 4 8x1 matrixes, 4 8x2 matrixes
        let mut mats = (0..4)
            .map(|_| RowMajorMatrix::<F>::rand(&mut rng, 8, 1))
            .collect_vec();
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 1,
        });
        mats.extend((0..4).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 8, 2)));
        let small_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 2,
        });

        let (commit, prover_data) = mmcs.commit(mats);

        // open the 3rd row of each matrix, mess with proof, and verify
        let mut batch_opening = mmcs.open_batch(3, &prover_data);
        batch_opening.opening_proof[0][0] += F::ONE;
        mmcs.verify_batch(
            &commit,
            &large_mat_dims.chain(small_mat_dims).collect_vec(),
            3,
            (&batch_opening).into(),
        )
        .expect_err("expected verification to fail");
    }

    #[test]
    fn size_gaps() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // 4 mats with 1000 rows, 8 columns
        let mut mats = (0..4)
            .map(|_| RowMajorMatrix::<F>::rand(&mut rng, 1000, 8))
            .collect_vec();
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 1000,
            width: 8,
        });

        // 5 mats with 125 rows, 8 columns
        mats.extend((0..5).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 125, 8)));
        let medium_mat_dims = (0..5).map(|_| Dimensions {
            height: 125,
            width: 8,
        });

        // 6 mats with 8 rows, 8 columns
        mats.extend((0..6).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 8, 8)));
        let small_mat_dims = (0..6).map(|_| Dimensions {
            height: 8,
            width: 8,
        });

        // 7 tiny mat with 1 row, 8 columns
        mats.extend((0..7).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 1, 8)));
        let tiny_mat_dims = (0..7).map(|_| Dimensions {
            height: 1,
            width: 8,
        });

        let dims = large_mat_dims
            .chain(medium_mat_dims)
            .chain(small_mat_dims)
            .chain(tiny_mat_dims)
            .collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);

        for &index in &[0, 6, 124, 999] {
            let batch_opening = mmcs.open_batch(index, &prover_data);
            mmcs.verify_batch(&commit, &dims, index, (&batch_opening).into())
                .expect("expected verification to succeed");
        }
    }

    #[test]
    fn different_widths() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<F>::rand(&mut rng, 32, i + 1))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let batch_opening = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, (&batch_opening).into())
            .expect("expected verification to succeed");
    }

    #[test]
    fn verify_rejects_wrong_row_width() {
        let mut rng = SmallRng::seed_from_u64(3);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        // Append one extra element to the opened row: 4 values become 5.
        let mut opening = mmcs.open_batch(3, &prover_data);
        opening.opened_values[0].push(F::ONE);

        let err = mmcs
            .verify_batch(&commit, &dims, 3, (&opening).into())
            .expect_err("row longer than the matrix width must be rejected");
        assert!(matches!(
            err,
            MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 4,
                got: 5,
            }
        ));
    }

    #[test]
    fn verify_rejects_shifted_row_boundary() {
        let mut rng = SmallRng::seed_from_u64(4);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // Two matrices of equal height hash into one flattened leaf stream:
        //
        //     leaf digest = hash(row_0[0..3] || row_1[0..5])
        //
        // Moving an element across the row boundary keeps the stream — and
        // hence the digest — identical, so only the width check can catch it.
        let mat_a = RowMajorMatrix::<F>::rand(&mut rng, 8, 3);
        let mat_b = RowMajorMatrix::<F>::rand(&mut rng, 8, 5);
        let dims = vec![mat_a.dimensions(), mat_b.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat_a, mat_b]);

        // Mutation: last element of row 0 becomes the first element of row 1.
        //
        //     row 0: [a, b, c]    → [a, b]
        //     row 1: [d, e, f, g, h] → [c, d, e, f, g, h]
        let mut opening = mmcs.open_batch(5, &prover_data);
        let moved = opening.opened_values[0].pop().unwrap();
        opening.opened_values[1].insert(0, moved);

        let err = mmcs
            .verify_batch(&commit, &dims, 5, (&opening).into())
            .expect_err("shifted row boundary must be rejected");
        assert!(matches!(
            err,
            MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 3,
                got: 2,
            }
        ));
    }

    #[test]
    fn cap_height_produces_shorter_proofs() {
        let mut rng = SmallRng::seed_from_u64(2);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // Tree with 32 leaves -> log2(32) = 5 layers
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = vec![Dimensions {
            height: 32,
            width: 4,
        }];

        for cap_height in 0..5 {
            let mmcs = MyMmcs::new(hash.clone(), compress.clone(), cap_height);
            let (commit, data) = mmcs.commit(vec![mat.clone()]);

            // Cap should have 2^cap_height elements
            assert_eq!(commit.num_roots(), 1 << cap_height);

            // Proof should have (log2(32) - cap_height) = 5 - cap_height elements
            let opening = mmcs.open_batch(17, &data);
            assert_eq!(opening.opening_proof.len(), 5 - cap_height);

            // Verification should succeed
            mmcs.verify_batch(&commit, &dims, 17, (&opening).into())
                .expect("verification should succeed");
        }
    }

    #[test]
    fn cap_verification_with_various_indices() {
        let mut rng = SmallRng::seed_from_u64(3);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mmcs = MyMmcs::new(hash, compress, 2); // cap_height = 2 -> 4 cap elements

        // 64 rows -> 6 layers, proofs have 6 - 2 = 4 siblings
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![Dimensions {
            height: 64,
            width: 8,
        }];

        let (commit, data) = mmcs.commit(vec![mat]);

        assert_eq!(commit.num_roots(), 4);

        // Test various indices that map to different cap entries
        for index in [0, 15, 16, 31, 32, 47, 48, 63] {
            let opening = mmcs.open_batch(index, &data);
            assert_eq!(opening.opening_proof.len(), 4);
            mmcs.verify_batch(&commit, &dims, index, (&opening).into())
                .unwrap_or_else(|_| panic!("verification at index {index} should succeed"));
        }
    }

    #[test]
    fn cap_tampered_proof_fails() {
        let mut rng = SmallRng::seed_from_u64(4);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mmcs = MyMmcs::new(hash, compress, 1);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let dims = vec![Dimensions {
            height: 16,
            width: 4,
        }];

        let (commit, data) = mmcs.commit(vec![mat]);

        let mut opening = mmcs.open_batch(7, &data);
        opening.opening_proof[0][0] += F::ONE;

        mmcs.verify_batch(&commit, &dims, 7, (&opening).into())
            .expect_err("tampered proof should fail verification");
    }

    #[test]
    fn cap_with_mixed_height_matrices() {
        let mut rng = SmallRng::seed_from_u64(5);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mmcs = MyMmcs::new(hash, compress, 2);

        // 64 rows and 8 rows (8 = 64 / 8, valid ratio)
        let mat_large = RowMajorMatrix::<F>::rand(&mut rng, 64, 4);
        let mat_small = RowMajorMatrix::<F>::rand(&mut rng, 8, 6);

        let dims = vec![
            Dimensions {
                height: 64,
                width: 4,
            },
            Dimensions {
                height: 8,
                width: 6,
            },
        ];

        let (commit, data) = mmcs.commit(vec![mat_large, mat_small]);
        assert_eq!(commit.num_roots(), 4);

        // Verify various indices
        for index in [0, 7, 8, 15, 32, 63] {
            let opening = mmcs.open_batch(index, &data);
            mmcs.verify_batch(&commit, &dims, index, (&opening).into())
                .unwrap_or_else(|_| panic!("verification at index {index} should succeed"));
        }
    }

    #[test]
    fn cap_height_shortened_for_small_trees() {
        let mut rng = SmallRng::seed_from_u64(6);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // For 4 rows: digest_layers has [4 row hashes, 2, 1] = 3 layers.
        // max cap_height = 3 - 1 = 2, giving 2^2 = 4 cap entries.
        // With cap_height = 10 (too large), it should be shortened to 2.
        let mmcs = MyMmcs::new(hash, compress, 10);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 4, 4);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 4 entries (shortened to height 2)
        assert_eq!(cap.num_roots(), 4);
        assert_eq!(cap.height(), 2);

        // Verify that opening and verification still work
        let (opening, proof) = mmcs.open_batch(0, &prover_data).unpack();
        let dims = vec![Dimensions {
            width: 4,
            height: 4,
        }];
        mmcs.verify_batch(&cap, &dims, 0, BatchOpeningRef::new(&opening, &proof))
            .unwrap();
    }

    #[test]
    fn single_row_matrix_with_cap_height() {
        let mut rng = SmallRng::seed_from_u64(7);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // Single-row matrix: digest_layers has [1 row hash] = 1 layer.
        // max cap_height = 1 - 1 = 0, so any cap_height > 0 gets shortened.
        let mmcs = MyMmcs::new(hash, compress, 5);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 1, 8);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 1 entry (shortened to height 0 = root)
        assert_eq!(cap.num_roots(), 1);
        assert_eq!(cap.height(), 0);

        // Proof length should be 0 (we're already at the cap)
        let (opening, proof) = mmcs.open_batch(0, &prover_data).unpack();
        assert!(
            proof.is_empty(),
            "proof should be empty for single-row tree"
        );

        let dims = vec![Dimensions {
            width: 8,
            height: 1,
        }];
        mmcs.verify_batch(&cap, &dims, 0, BatchOpeningRef::new(&opening, &proof))
            .unwrap();
    }

    #[test]
    fn proof_length_zero_at_max_cap_height() {
        let mut rng = SmallRng::seed_from_u64(8);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // 8 rows: digest_layers has [8, 4, 2, 1] = 4 layers, log_max_height = 3.
        // max cap_height = 4 - 1 = 3, giving 2^3 = 8 cap entries.
        // With cap_height = 3, proof length = 3 - 3 = 0.
        let mmcs = MyMmcs::new(hash, compress, 3);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 8 entries (all leaf hashes)
        assert_eq!(cap.num_roots(), 8);
        assert_eq!(cap.height(), 3);

        // Proof length should be 0 since cap is at leaf level
        for index in 0..8 {
            let (opening, proof) = mmcs.open_batch(index, &prover_data).unpack();
            assert!(
                proof.is_empty(),
                "proof should be empty when cap is at leaf level"
            );

            let dims = vec![Dimensions {
                width: 4,
                height: 8,
            }];
            mmcs.verify_batch(&cap, &dims, index, BatchOpeningRef::new(&opening, &proof))
                .unwrap();
        }
    }

    #[test]
    fn cap_height_exact_boundary() {
        let mut rng = SmallRng::seed_from_u64(9);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // 16 rows: digest_layers has [16, 8, 4, 2, 1] = 5 layers, log_max_height = 4.
        // max cap_height = 5 - 1 = 4.
        // Test cap_height = 4 (exact boundary, no shortening).
        let mmcs = MyMmcs::new(hash, compress, 4);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 16 entries
        assert_eq!(cap.num_roots(), 16);
        assert_eq!(cap.height(), 4);

        // Proof length should be 0
        let (opening, proof) = mmcs.open_batch(0, &prover_data).unpack();
        assert!(proof.is_empty());

        let dims = vec![Dimensions {
            width: 4,
            height: 16,
        }];
        mmcs.verify_batch(&cap, &dims, 0, BatchOpeningRef::new(&opening, &proof))
            .unwrap();

        // Also test cap_height = 3 (one below boundary, proof_len = 1)
        let mmcs2 = MyMmcs::new(
            MyHash::new(Perm::new_from_rng_128(&mut rng)),
            MyCompress::new(Perm::new_from_rng_128(&mut rng)),
            3,
        );
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let (cap2, prover_data2) = mmcs2.commit(vec![mat2]);

        assert_eq!(cap2.num_roots(), 8);
        assert_eq!(cap2.height(), 3);

        let (opening2, proof2) = mmcs2.open_batch(0, &prover_data2).unpack();
        assert_eq!(proof2.len(), 1, "proof should have 1 sibling");

        mmcs2
            .verify_batch(&cap2, &dims, 0, BatchOpeningRef::new(&opening2, &proof2))
            .unwrap();
    }
}
