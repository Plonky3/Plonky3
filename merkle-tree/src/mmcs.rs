use alloc::vec::Vec;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::PackedField;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Dimensions, Matrix, MatrixRows};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;

use crate::FieldMerkleTree;

/// A vector commitment scheme backed by a `FieldMerkleTree`.
///
/// Generics:
/// - `P`: a leaf value TODO
/// - `H`: the leaf hasher
/// - `C`: the digest compression function
#[derive(Copy, Clone)]
pub struct FieldMerkleTreeMmcs<P, H, C, const DIGEST_ELEMS: usize> {
    hash: H,
    compress: C,
    _phantom_p: PhantomData<P>,
}

impl<P, H, C, const DIGEST_ELEMS: usize> FieldMerkleTreeMmcs<P, H, C, DIGEST_ELEMS> {
    pub fn new(hash: H, compress: C) -> Self {
        Self {
            hash,
            compress,
            _phantom_p: PhantomData,
        }
    }
}

impl<P, H, C, const DIGEST_ELEMS: usize> Mmcs<P::Scalar>
    for FieldMerkleTreeMmcs<P, H, C, DIGEST_ELEMS>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    type ProverData = FieldMerkleTree<P::Scalar, DIGEST_ELEMS>;
    type Commitment = [P::Scalar; DIGEST_ELEMS];
    type Proof = Vec<[P::Scalar; DIGEST_ELEMS]>;
    type Error = ();
    type Mat<'a> = RowMajorMatrixView<'a, P::Scalar> where H: 'a, C: 'a;

    fn open_batch(
        &self,
        index: usize,
        prover_data: &FieldMerkleTree<P::Scalar, DIGEST_ELEMS>,
    ) -> (Vec<Vec<P::Scalar>>, Vec<[P::Scalar; DIGEST_ELEMS]>) {
        let max_height = self.get_max_height(prover_data);
        let log_max_height = log2_ceil_usize(max_height);

        let openings = prover_data
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.row(reduced_index).collect()
            })
            .collect_vec();

        let proof = (0..log_max_height)
            .map(|i| prover_data.digest_layers[i][(index >> i) ^ 1])
            .collect();

        (openings, proof)
    }

    fn get_matrices<'a>(
        &'a self,
        prover_data: &'a Self::ProverData,
    ) -> Vec<RowMajorMatrixView<'a, P::Scalar>> {
        prover_data.leaves.iter().map(|mat| mat.as_view()).collect()
    }

    fn verify_batch(
        &self,
        commit: &[P::Scalar; DIGEST_ELEMS],
        dimensions: &[Dimensions],
        mut index: usize,
        opened_values: &[Vec<P::Scalar>],
        proof: &Vec<[P::Scalar; DIGEST_ELEMS]>,
    ) -> Result<(), Self::Error> {
        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        let mut curr_height_padded = heights_tallest_first
            .peek()
            .unwrap()
            .1
            .height
            .next_power_of_two();

        let mut root = self.hash.hash_iter_slices(
            heights_tallest_first
                .peeking_take_while(|(_, dims)| {
                    dims.height.next_power_of_two() == curr_height_padded
                })
                .map(|(i, _)| opened_values[i].as_slice()),
        );

        for &sibling in proof.iter() {
            let (left, right) = if index & 1 == 0 {
                (root, sibling)
            } else {
                (sibling, root)
            };

            root = self.compress.compress([left, right]);
            index >>= 1;
            curr_height_padded >>= 1;

            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == curr_height_padded);
            if let Some(next_height) = next_height {
                let next_height_openings_digest = self.hash.hash_iter_slices(
                    heights_tallest_first
                        .peeking_take_while(|(_, dims)| dims.height == next_height)
                        .map(|(i, _)| opened_values[i].as_slice()),
                );

                root = self.compress.compress([root, next_height_openings_digest]);
            }
        }

        if root == *commit {
            Ok(())
        } else {
            Err(())
        }
    }
}

impl<P, H, C, const DIGEST_ELEMS: usize> DirectMmcs<P::Scalar>
    for FieldMerkleTreeMmcs<P, H, C, DIGEST_ELEMS>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    fn commit(
        &self,
        inputs: Vec<RowMajorMatrix<P::Scalar>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let tree = FieldMerkleTree::new::<P, H, C>(&self.hash, &self.compress, inputs);
        let root = tree.root();
        (root, tree)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_commit::{DirectMmcs, Mmcs};
    use p3_field::{AbstractField, Field};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};
    use p3_mds::coset_mds::CosetMds;
    use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
    use p3_symmetric::{
        CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
    };
    use rand::thread_rng;

    use crate::FieldMerkleTreeMmcs;

    type F = BabyBear;

    type MyMds = CosetMds<F, 16>;
    type Perm = Poseidon2<F, MyMds, DiffusionMatrixBabybear, 16, 5>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs = FieldMerkleTreeMmcs<<F as Field>::Packing, MyHash, MyCompress, 8>;

    #[test]
    fn commit_single_1x8() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());

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
        assert_eq!(commit, expected_result);
    }

    #[test]
    fn commit_single_2x2() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());

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
        assert_eq!(commit, expected_result);
    }

    #[test]
    fn commit_single_2x3() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());
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
        assert_eq!(commit, expected_result);
    }

    #[test]
    fn commit_mixed() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());
        let default_digest = [F::ZERO; 8];

        // mat_1 = [
        //   0 1
        //   2 1
        //   2 2
        // ]
        let mat_1 = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE, F::TWO, F::TWO], 2);
        // mat_2 = [
        //   1 2 1
        //   0 2 2
        // ]
        let mat_2 = RowMajorMatrix::new(vec![F::ONE, F::TWO, F::ONE, F::ZERO, F::TWO, F::TWO], 3);

        let (commit, prover_data) = mmcs.commit(vec![mat_1, mat_2]);

        let mat_1_leaf_hashes = [
            hash.hash_slice(&[F::ZERO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::TWO]),
        ];
        let mat_2_leaf_hashes = [
            hash.hash_slice(&[F::ONE, F::TWO, F::ONE]),
            hash.hash_slice(&[F::ZERO, F::TWO, F::TWO]),
        ];

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([mat_1_leaf_hashes[0], mat_1_leaf_hashes[1]]),
                mat_2_leaf_hashes[0],
            ]),
            compress.compress([
                compress.compress([mat_1_leaf_hashes[2], default_digest]),
                mat_2_leaf_hashes[1],
            ]),
        ]);
        assert_eq!(commit, expected_result);

        let (opened_values, _proof) = mmcs.open_batch(2, &prover_data);
        assert_eq!(
            opened_values,
            vec![vec![F::TWO, F::TWO], vec![F::ZERO, F::TWO, F::TWO]]
        );
    }

    #[test]
    fn commit_either_order() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        let input_1 = RowMajorMatrix::<F>::rand(&mut rng, 5, 8);
        let input_2 = RowMajorMatrix::<F>::rand(&mut rng, 3, 16);

        let (commit_1_2, _) = mmcs.commit(vec![input_1.clone(), input_2.clone()]);
        let (commit_2_1, _) = mmcs.commit(vec![input_2, input_1]);
        assert_eq!(commit_1_2, commit_2_1);
    }

    #[test]
    #[should_panic]
    fn mismatched_heights() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // attempt to commit to a mat with 8 rows and a mat with 7 rows. this should panic.
        let large_mat = RowMajorMatrix::new(
            [1, 2, 3, 4, 5, 6, 7, 8].map(F::from_canonical_u8).to_vec(),
            1,
        );
        let small_mat =
            RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7].map(F::from_canonical_u8).to_vec(), 1);
        let _ = mmcs.commit(vec![large_mat, small_mat]);
    }

    #[test]
    fn verify_tampered_proof_fails() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // 4 8x1 matrixes, 4 8x2 matrixes
        let large_mats = (0..4).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 8, 1));
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 1,
        });
        let small_mats = (0..4).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 8, 2));
        let small_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 2,
        });

        let (commit, prover_data) = mmcs.commit(large_mats.chain(small_mats).collect_vec());

        // open the 3rd row of each matrix, mess with proof, and verify
        let (opened_values, mut proof) = mmcs.open_batch(3, &prover_data);
        proof[0][0] += F::ONE;
        mmcs.verify_batch(
            &commit,
            &large_mat_dims.chain(small_mat_dims).collect_vec(),
            3,
            &opened_values,
            &proof,
        )
        .expect_err("expected verification to fail");
    }

    #[test]
    fn size_gaps() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // 4 mats with 1000 rows, 8 columns
        let large_mats = (0..4).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 1000, 8));
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 1000,
            width: 8,
        });

        // 5 mats with 70 rows, 8 columns
        let medium_mats = (0..5).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 70, 8));
        let medium_mat_dims = (0..5).map(|_| Dimensions {
            height: 70,
            width: 8,
        });

        // 6 mats with 8 rows, 8 columns
        let small_mats = (0..6).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 8, 8));
        let small_mat_dims = (0..6).map(|_| Dimensions {
            height: 8,
            width: 8,
        });

        let (commit, prover_data) = mmcs.commit(
            large_mats
                .chain(medium_mats)
                .chain(small_mats)
                .collect_vec(),
        );

        // open the 6th row of each matrix and verify
        let (opened_values, proof) = mmcs.open_batch(6, &prover_data);
        mmcs.verify_batch(
            &commit,
            &large_mat_dims
                .chain(medium_mat_dims)
                .chain(small_mat_dims)
                .collect_vec(),
            6,
            &opened_values,
            &proof,
        )
        .expect("expected verification to succeed");
    }

    #[test]
    fn different_widths() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<F>::rand(&mut thread_rng(), 32, i + 1))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let (opened_values, proof) = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, &opened_values, &proof)
            .expect("expected verification to succeed");
    }
}
