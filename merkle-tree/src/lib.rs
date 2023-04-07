#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::marker::PhantomData;
use itertools::Itertools;
use p3_commit::oracle::{ConcreteOracle, Dimensions, Oracle};
use p3_field::matrix::dense::DenseMatrix;
use p3_field::matrix::Matrix;
use p3_symmetric::compression::CompressionFunction;
use p3_symmetric::hasher::IterHasher;

// TODO: Add Jaqui's cache-friendly version, maybe as a separate alternative impl.

// TODO: Add a variant that supports pruning overlapping paths?
// How would we keep track of previously-seen paths - make the Oracle methods take &mut self?

/// A binary Merkle tree, with leaves of type `L` and digests of type `D`.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an `Oracle`,
/// see `MerkleTreeOracle`.
pub struct MerkleTree<L, D> {
    leaves: Vec<DenseMatrix<L>>,
    digest_layers: Vec<Vec<D>>,
}

impl<L, D> MerkleTree<L, D> {
    pub fn new<H, C>(leaves: Vec<DenseMatrix<L>>) -> Self
    where
        for<'a> H: IterHasher<&'a L, D>,
        C: CompressionFunction<D, 2>,
    {
        assert!(!leaves.is_empty(), "No matrices given?");
        for leaf in &leaves {
            assert!(
                leaf.height().is_power_of_two(),
                "Matrix height not a power of two"
            )
        }

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();
        let max_height = leaves_largest_first.peek().unwrap().height();

        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let first_digest_layer = (0..max_height)
            .map(|i| H::hash_iter(tallest_matrices.iter().flat_map(|m| m.row(i).iter())))
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer];
        loop {
            let prev_layer = digest_layers
                .last()
                .map(|v| v.as_slice())
                .unwrap_or_default();
            if prev_layer.len() == 1 {
                break;
            }

            // The matrices that get inserted at this layer.
            let tallest_matrices = leaves_largest_first
                .peeking_take_while(|m| m.height() == prev_layer.len())
                .collect_vec();

            let next_len = prev_layer.len() >> 1;
            let mut next_digests = Vec::with_capacity(next_len);
            for i in 0..next_len {
                let left = &prev_layer[2 * i];
                let right = &prev_layer[2 * i + 1];
                let mut digest = C::compress(&[left, right]);
                if !tallest_matrices.is_empty() {
                    let tallest_digest =
                        H::hash_iter(tallest_matrices.iter().flat_map(|m| m.row(i).iter()));
                    digest = C::compress(&[&digest, &tallest_digest]);
                }
                next_digests.push(digest);
            }

            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digest_layers,
        }
    }

    pub fn root(&self) -> D
    where
        D: Clone,
    {
        self.digest_layers.last().unwrap()[0].clone()
    }
}

/// A vector commitment scheme backed by a Merkle tree.
///
/// Generics:
/// - `L`: a leaf value
/// - `D`: a digest
/// - `H`: the leaf hasher
/// - `C`: the digest compression function
pub struct MerkleTreeOracle<L, D, H, C>
where
    for<'a> H: IterHasher<&'a L, D>,
    C: CompressionFunction<D, 2>,
{
    _phantom_l: PhantomData<L>,
    _phantom_d: PhantomData<D>,
    _phantom_h: PhantomData<H>,
    _phantom_c: PhantomData<C>,
}

impl<L, D, H, C> Oracle<L> for MerkleTreeOracle<L, D, H, C>
where
    L: Clone,
    for<'a> H: IterHasher<&'a L, D>,
    C: CompressionFunction<D, 2>,
{
    type ProverData = MerkleTree<L, D>;
    type Commitment = D;
    type Proof = Vec<D>;
    type Error = ();

    fn open_batch(row: usize, prover_data: &MerkleTree<L, D>) -> (Vec<&[L]>, Vec<D>) {
        let leaf = prover_data
            .leaves
            .iter()
            .map(|matrix| matrix.row(row))
            .collect();
        let proof = vec![]; // TODO
        (leaf, proof)
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

impl<L, D, H, C> ConcreteOracle<L> for MerkleTreeOracle<L, D, H, C>
where
    L: Clone,
    D: Clone,
    for<'a> H: IterHasher<&'a L, D>,
    C: CompressionFunction<D, 2>,
{
    fn commit_batch(inputs: Vec<DenseMatrix<L>>) -> (Self::ProverData, Self::Commitment) {
        let tree = MerkleTree::new::<H, C>(inputs);
        let root = tree.root();
        (tree, root)
    }
}
