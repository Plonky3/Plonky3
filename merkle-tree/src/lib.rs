#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::iter;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::{Dimensions, DirectMmcs, Mmcs};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Matrix, MatrixRows};
use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::hasher::CryptographicHasher;
use p3_util::{log2_ceil_usize, log2_strict_usize};

// TODO: Add a variant that supports pruning overlapping paths?
// How would we keep track of previously-seen paths - make the MMCS methods take &mut self?

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
    pub fn new<H, C>(h: &H, c: &C, leaves: Vec<RowMajorMatrix<L>>) -> Self
    where
        L: Copy,
        D: Copy + Default,
        H: CryptographicHasher<L, D>,
        C: PseudoCompressionFunction<D, 2>,
    {
        assert!(!leaves.is_empty(), "No matrices given?");

        // TODO: Check the matching height condition.

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();
        let max_height = leaves_largest_first.peek().unwrap().height();
        let max_height_padded = log2_ceil_usize(max_height);

        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let first_digest_layer = (0..max_height)
            .map(|i| h.hash_iter(tallest_matrices.iter().flat_map(|m| m.row(i))))
            .chain(iter::repeat(D::default()))
            .take(max_height_padded)
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer];
        loop {
            let prev_layer = digest_layers.last().map(Vec::as_slice).unwrap_or_default();
            if prev_layer.len() == 1 {
                break;
            }

            // The matrices that get inserted at this layer.
            let tallest_matrices = leaves_largest_first
                .peeking_take_while(|m| log2_ceil_usize(m.height()) == prev_layer.len())
                .collect_vec();

            let next_digests = compression_layer(prev_layer, tallest_matrices, h, c);
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
fn compression_layer<L, D, H, C, Mat>(
    prev_layer: &[D],
    tallest_matrices: Vec<&Mat>,
    h: &H,
    c: &C,
) -> Vec<D>
where
    L: Copy,
    D: Copy + Default,
    H: CryptographicHasher<L, D>,
    C: PseudoCompressionFunction<D, 2>,
    Mat: MatrixRows<L>,
{
    let next_len_padded = prev_layer.len() >> 1;
    let mut next_digests = Vec::with_capacity(next_len_padded);

    if tallest_matrices.is_empty() {
        for i in 0..next_len_padded {
            let left = prev_layer[2 * i];
            let right = prev_layer[2 * i + 1];
            let digest = c.compress([left, right]);
            next_digests.push(digest);
        }
        return next_digests;
    }

    let next_len = tallest_matrices[0].height();
    for i in 0..next_len {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let mut digest = c.compress([left, right]);
        let tallest_digest =
            h.hash_iter(tallest_matrices.iter().flat_map(|m| m.row(i).into_iter()));
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
    D: Clone,
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
        let log_max_height = log2_strict_usize(max_height);

        let leaf = prover_data
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_strict_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.row(reduced_index).collect()
            })
            .collect();
        let proof = vec![]; // TODO
        (leaf, proof)
    }

    fn get_matrices<'a>(
        &'a self,
        prover_data: &'a Self::ProverData,
    ) -> Vec<RowMajorMatrixView<'a, L>> {
        prover_data.leaves.iter().map(|mat| mat.as_view()).collect()
    }

    fn verify_batch(
        &self,
        _commit: &D,
        _dimensions: &[Dimensions],
        _index: usize,
        _opened_values: Vec<Vec<L>>,
        _proof: &Vec<D>,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}

impl<L, D, H, C> DirectMmcs<L> for MerkleTreeMmcs<L, D, H, C>
where
    L: 'static + Copy,
    D: Copy + Default,
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

    use p3_commit::{DirectMmcs, Mmcs};
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_matrix::dense::RowMajorMatrix;
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
}
