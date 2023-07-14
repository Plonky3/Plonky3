#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::iter;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::{Dimensions, DirectMMCS, MMCS};
use p3_matrix::{Matrix, MatrixRows};
use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::hasher::CryptographicHasher;
use p3_util::log2_ceil_usize;

// TODO: Add a variant that supports pruning overlapping paths?
// How would we keep track of previously-seen paths - make the MMCS methods take &mut self?

/// A binary Merkle tree, with leaves of type `L` and digests of type `D`.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an `MMCS`,
/// see `MerkleTreeMMCS`.
pub struct MerkleTree<L, D, Mat: Matrix<L>> {
    leaves: Vec<Mat>,
    digest_layers: Vec<Vec<D>>,
    _phantom_l: PhantomData<L>,
}

impl<L, D, Mat> MerkleTree<L, D, Mat>
where
    Mat: for<'a> MatrixRows<'a, L>,
{
    /// Matrix heights need not be powers of two. However, if the heights of two given matrices
    /// round up to the same power of two, they must be equal.
    pub fn new<H, C>(h: &H, c: &C, leaves: Vec<Mat>) -> Self
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
            .map(|i| {
                h.hash_iter(
                    tallest_matrices
                        .iter()
                        .map(|m| m.row(i).into_iter().copied())
                        .flatten(),
                )
            })
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
    Mat: for<'a> MatrixRows<'a, L>,
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
        let tallest_digest = h.hash_iter(
            tallest_matrices
                .iter()
                .map(|m| m.row(i).into_iter().copied())
                .flatten(),
        );
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
pub struct MerkleTreeMMCS<L, D, H, C, Mat> {
    hash: H,
    compress: C,
    _phantom_l: PhantomData<L>,
    _phantom_d: PhantomData<D>,
    _phantom_mat: PhantomData<Mat>,
}

impl<L, D, H, C, Mat> MerkleTreeMMCS<L, D, H, C, Mat> {
    pub fn new(hash: H, compress: C) -> Self {
        Self {
            hash,
            compress,
            _phantom_l: PhantomData,
            _phantom_d: PhantomData,
            _phantom_mat: PhantomData,
        }
    }
}

impl<L, D, H, C, Mat> MMCS<L> for MerkleTreeMMCS<L, D, H, C, Mat>
where
    L: Clone,
    H: CryptographicHasher<L, D>,
    C: PseudoCompressionFunction<D, 2>,
    Mat: for<'a> MatrixRows<'a, L>,
{
    type ProverData = MerkleTree<L, D, Mat>;
    type Commitment = D;
    type Proof = Vec<D>;
    type Error = ();
    type Mat = Mat;

    fn open_batch(row: usize, prover_data: &MerkleTree<L, D, Mat>) -> (Vec<Vec<L>>, Vec<D>) {
        let leaf = prover_data
            .leaves
            .iter()
            .map(|matrix| matrix.row(row).into_iter().cloned().collect())
            .collect();
        let proof = vec![]; // TODO
        (leaf, proof)
    }

    fn get_matrices(prover_data: &Self::ProverData) -> &[Mat] {
        &prover_data.leaves
    }

    fn verify_batch(
        _commit: &D,
        _dimensions: &[Dimensions],
        _index: usize,
        _item: Vec<L>,
        _proof: &Vec<D>,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}

impl<L, D, H, C, Mat> DirectMMCS<L> for MerkleTreeMMCS<L, D, H, C, Mat>
where
    L: Copy,
    D: Copy + Default,
    H: CryptographicHasher<L, D>,
    C: PseudoCompressionFunction<D, 2>,
    Mat: for<'a> MatrixRows<'a, L>,
{
    fn commit(&self, inputs: Vec<Mat>) -> (Self::Commitment, Self::ProverData) {
        let tree = MerkleTree::new(&self.hash, &self.compress, inputs);
        let root = tree.root();
        (root, tree)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_commit::DirectMMCS;
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::compression::TruncatedPermutation;
    use rand::thread_rng;

    use crate::MerkleTreeMMCS;

    #[test]
    fn commit() {
        use p3_keccak::KeccakF;

        type C = TruncatedPermutation<u8, KeccakF, 2, 32, 200>;
        let compress = C::new(KeccakF);

        type Mmcs = MerkleTreeMMCS<u8, [u8; 32], Keccak256Hash, C, RowMajorMatrix<u8>>;
        let mmcs = Mmcs::new(Keccak256Hash, compress);

        let mut rng = thread_rng();

        // First try a power-of-two height.
        let mat = RowMajorMatrix::rand(&mut rng, 256, 13);
        mmcs.commit(vec![mat]);

        // Then a non-power-of-two height.
        let mat = RowMajorMatrix::rand(&mut rng, 200, 13);
        mmcs.commit(vec![mat]);
    }
}
