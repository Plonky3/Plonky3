extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::iter;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::{DirectMmcs, Mmcs};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Dimensions, Matrix, MatrixRowSlices, MatrixRows};
use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::hasher::CryptographicHasher;
use p3_util::log2_ceil_usize;
use tracing::instrument;

/// A binary Merkle tree, with leaves of type `L` and digests of type `D`.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `MerkleTreeMmcs`.
pub struct MerkleTree<L, D> {
    leaves: Vec<RowMajorMatrix<L>>,
    digest_layers: Vec<Vec<D>>,
    _phantom_l: PhantomData<L>,
}

impl<L, D> MerkleTree<L, D> {
    /// Matrix heights need not be powers of two. However, if the heights of two given matrices
    /// round up to the same power of two, they must be equal.
    #[instrument(name = "build merkle tree", level = "debug", skip_all,
                 fields(dimensions = alloc::format!("{:?}", leaves.iter().map(|l| l.dimensions()).collect::<Vec<_>>())))]
    pub fn new<H, C>(h: &H, c: &C, leaves: Vec<RowMajorMatrix<L>>) -> Self
    where
        L: Copy,
        D: Copy + Default + PartialEq,
        H: CryptographicHasher<L, D>,
        C: PseudoCompressionFunction<D, 2>,
    {
        assert!(!leaves.is_empty(), "No matrices given?");

        // check height property
        assert!(
            leaves
                .iter()
                .map(|m| m.height())
                .sorted_by_key(|&h| Reverse(h))
                .tuple_windows()
                .all(|(curr, next)| curr == next
                    || curr.next_power_of_two() != next.next_power_of_two()),
            "matrix heights that round up to the same power of two must be equal"
        );

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();
        let max_height = leaves_largest_first.peek().unwrap().height();
        let max_height_padded = max_height.next_power_of_two();

        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let first_digest_layer = (0..max_height)
            .map(|i| h.hash_iter_slices(tallest_matrices.iter().map(|m| m.row_slice(i))))
            .chain(iter::repeat(D::default()))
            .take(max_height_padded)
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer];
        loop {
            let prev_layer = digest_layers.last().map(Vec::as_slice).unwrap_or_default();
            if prev_layer.len() == 1 {
                break;
            }
            let next_layer_len = prev_layer.len() / 2;

            // The matrices that get injected at this layer.
            let matrices_to_inject = leaves_largest_first
                .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                .collect_vec();

            let next_digests = compression_layer(prev_layer, matrices_to_inject, h, c);
            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digest_layers,
            _phantom_l: PhantomData,
        }
    }

    #[must_use]
    pub fn root(&self) -> D
    where
        D: Clone,
    {
        self.digest_layers.last().unwrap()[0].clone()
    }
}

/// Compress `n` digests from the previous layer into `n/2` digests, while potentially mixing in
/// some leaf data, if there are input matrices with (padded) height `n/2`.
fn compression_layer<L, D, H, C>(
    prev_layer: &[D],
    matrices_to_inject: Vec<&RowMajorMatrix<L>>,
    h: &H,
    c: &C,
) -> Vec<D>
where
    L: Copy,
    D: Copy + Default + PartialEq,
    H: CryptographicHasher<L, D>,
    C: PseudoCompressionFunction<D, 2>,
{
    let next_len_padded = prev_layer.len() / 2;
    let mut next_digests = Vec::with_capacity(next_len_padded);

    if matrices_to_inject.is_empty() {
        for i in 0..next_len_padded {
            let left = prev_layer[2 * i];
            let right = prev_layer[2 * i + 1];
            let digest = c.compress([left, right]);
            next_digests.push(digest);
        }
        return next_digests;
    }

    let next_len = matrices_to_inject[0].height();
    for i in 0..next_len {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let mut digest = c.compress([left, right]);
        let tallest_digest = h.hash_iter_slices(matrices_to_inject.iter().map(|m| m.row_slice(i)));
        digest = c.compress([digest, tallest_digest]);
        next_digests.push(digest);
    }
    for i in next_len..next_len_padded {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let mut digest = c.compress([left, right]);
        let tallest_digest = D::default();
        digest = c.compress([digest, tallest_digest]);
        next_digests.push(digest);
    }
    next_digests
}

/// A vector commitment scheme backed by a Merkle tree.
///
/// Generics:
/// - `L`: a leaf value
/// - `D`: a digest
/// - `H`: the leaf hasher
/// - `C`: the digest compression function
#[derive(Copy, Clone)]
pub struct MerkleTreeMmcs<L, D, H, C> {
    hash: H,
    compress: C,
    _phantom_l: PhantomData<L>,
    _phantom_d: PhantomData<D>,
}

impl<L, D, H, C> MerkleTreeMmcs<L, D, H, C> {
    pub fn new(hash: H, compress: C) -> Self {
        Self {
            hash,
            compress,
            _phantom_l: PhantomData,
            _phantom_d: PhantomData,
        }
    }
}

impl<L, D, H, C> Mmcs<L> for MerkleTreeMmcs<L, D, H, C>
where
    L: 'static + Clone,
    D: Copy + Default + PartialEq,
    H: CryptographicHasher<L, D>,
    C: PseudoCompressionFunction<D, 2>,
{
    type ProverData = MerkleTree<L, D>;
    type Commitment = D;
    type Proof = Vec<D>;
    type Error = ();
    type Mat<'a> = RowMajorMatrixView<'a, L> where D: 'a, H: 'a, C: 'a;

    fn open_batch(&self, index: usize, prover_data: &MerkleTree<L, D>) -> (Vec<Vec<L>>, Vec<D>) {
        let max_height = self.get_max_height(prover_data);
        let log_max_height = log2_ceil_usize(max_height);

        // get the the `j`th row of each matrix `M[i]`,
        // where `j = index >> (log2_ceil(max_height) - log2_ceil(M[i].height))`.
        let openings: Vec<Vec<L>> = prover_data
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.row(reduced_index).collect()
            })
            .collect();

        let proof = (0..log_max_height)
            .map(|i| prover_data.digest_layers[i][(index >> i) ^ 1].clone())
            .collect();

        (openings, proof)
    }

    fn get_matrices<'a>(
        &'a self,
        prover_data: &'a Self::ProverData,
    ) -> Vec<RowMajorMatrixView<'a, L>> {
        prover_data.leaves.iter().map(|mat| mat.as_view()).collect()
    }

    fn verify_batch(
        &self,
        commit: &D,
        dimensions: &[Dimensions],
        mut index: usize,
        opened_values: &[Vec<L>],
        proof: &Vec<D>,
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

impl<L, D, H, C> DirectMmcs<L> for MerkleTreeMmcs<L, D, H, C>
where
    L: 'static + Copy,
    D: Copy + Default + PartialEq,
    H: CryptographicHasher<L, D>,
    C: PseudoCompressionFunction<D, 2>,
{
    fn commit(&self, inputs: Vec<RowMajorMatrix<L>>) -> (Self::Commitment, Self::ProverData) {
        let tree = MerkleTree::new(&self.hash, &self.compress, inputs);
        let root = tree.root();
        (root, tree)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_commit::{DirectMmcs, Mmcs};
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_matrix::{dense::RowMajorMatrix, Dimensions, Matrix};
    use p3_symmetric::compression::TruncatedPermutation;
    use rand::thread_rng;

    use crate::MerkleTreeMmcs;

    #[test]
    fn commit() {
        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMmcs<u8, [u8; 32], Keccak256Hash, C>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        let mut rng = thread_rng();

        // First try a power-of-two height.
        let mat = RowMajorMatrix::rand(&mut rng, 256, 13);
        mmcs.commit(vec![mat]);

        // Then a non-power-of-two height.
        let mat = RowMajorMatrix::rand(&mut rng, 200, 13);
        mmcs.commit(vec![mat]);
    }

    #[test]
    fn open() {
        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMmcs<u8, [u8; 32], Keccak256Hash, C>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        // large_mat has 8 rows and 1 col; small_mat has 4 rows and 2 cols.
        let large_mat = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        let small_mat = RowMajorMatrix::new(vec![10, 11, 20, 21, 30, 31, 40, 41], 2);
        let (_commit, prover_data) = mmcs.commit(vec![large_mat, small_mat]);

        let (opened_values, _proof) = mmcs.open_batch(3, &prover_data);
        assert_eq!(opened_values, vec![vec![4], vec![20, 21]]);
    }

    #[test]
    #[should_panic]
    fn height_property() {
        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMmcs<u8, [u8; 32], Keccak256Hash, C>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        // attempt to commit to a mat with 8 rows and a mat with 7 rows. this should panic
        let large_mat = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        let small_mat = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7], 1);
        let _ = mmcs.commit(vec![large_mat, small_mat]);
    }

    #[test]
    fn verify() {
        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMmcs<u8, [u8; 32], Keccak256Hash, C>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        // large_mat has 8 rows and 1 col; small_mat has 4 rows and 2 cols.
        let large_mat = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        let large_mat_dims = large_mat.dimensions();
        let small_mat = RowMajorMatrix::new(vec![10, 11, 20, 21, 30, 31, 40, 41], 2);
        let small_mat_dims = small_mat.dimensions();
        let (commit, prover_data) = mmcs.commit(vec![large_mat, small_mat]);

        let (opened_values, proof) = mmcs.open_batch(3, &prover_data);
        assert_eq!(opened_values, vec![vec![4], vec![20, 21]]);

        mmcs.verify_batch(
            &commit,
            &[large_mat_dims, small_mat_dims],
            3,
            &opened_values,
            &proof,
        )
        .expect("expected verification to succeed");
    }

    #[test]
    fn verify_tampered_proof_fails() {
        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMmcs<u8, [u8; 32], Keccak256Hash, C>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        // 4 8x1 matrixes, 4 8x2 matrixes
        let large_mats = (0..4).map(|_| RowMajorMatrix::<u8>::rand(&mut thread_rng(), 8, 1));
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 1,
        });
        let small_mats = (0..4).map(|_| RowMajorMatrix::<u8>::rand(&mut thread_rng(), 8, 2));
        let small_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 2,
        });

        let (commit, prover_data) = mmcs.commit(large_mats.chain(small_mats).collect_vec());

        // open the 3rd row of each matrix, mess with proof, and verify
        let (opened_values, mut proof) = mmcs.open_batch(3, &prover_data);
        proof[0][0] ^= 1;
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
        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMmcs<u8, [u8; 32], Keccak256Hash, C>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        // 4 mats with 1024 rows, 8 columns
        let large_mats = (0..4).map(|_| RowMajorMatrix::<u8>::rand(&mut thread_rng(), 1024, 8));
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 1 << 10,
            width: 8,
        });

        // 5 mats with 64 rows, 8 columns
        let medium_mats = (0..5).map(|_| RowMajorMatrix::<u8>::rand(&mut thread_rng(), 64, 8));
        let medium_mat_dims = (0..5).map(|_| Dimensions {
            height: 1 << 6,
            width: 8,
        });

        // 6 mats with 8 rows, 8 columns
        let small_mats = (0..6).map(|_| RowMajorMatrix::<u8>::rand(&mut thread_rng(), 8, 8));
        let small_mat_dims = (0..6).map(|_| Dimensions {
            height: 1 << 3,
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
        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMmcs<u8, [u8; 32], Keccak256Hash, C>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<u8>::rand(&mut thread_rng(), 32, i + 1))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let (opened_values, proof) = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, &opened_values, &proof)
            .expect("expected verification to succeed");
    }
}
