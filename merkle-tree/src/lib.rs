#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use itertools::Itertools;
use p3_commit::oracle::{ConcreteOracle, Oracle};
use p3_symmetric::compression::CompressionFunction;
use p3_symmetric::hasher::CryptographicHasher;

// TODO: Add Jaqui's cache-friendly version, maybe as a separate alternative impl.

// TODO: Add a variant that supports pruning overlapping paths?
// How would we keep track of previously-seen paths - make the Oracle methods take &mut self?

/// A standard binary Merkle tree.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an `Oracle`, see `MerkleTreeVCS`.
pub struct MerkleTree<L, D> {
    pub leaves: Vec<L>,
    pub digest_layers: Vec<Vec<D>>,
}

impl<L, D> MerkleTree<L, D> {
    pub fn new<H, C>(leaves: Vec<L>) -> Self
    where
        H: CryptographicHasher<L, D>,
        C: CompressionFunction<D, 2>,
    {
        // TODO: Assert that leaves.len() is a power of 2.
        let leaf_digests = leaves.iter().map(|l| H::hash(l)).collect_vec();
        let mut digest_layers = vec![leaf_digests];
        while digest_layers.last().unwrap().len() > 1 {
            let next_digests = digest_layers
                .last()
                .unwrap()
                .iter()
                .tuples()
                .map(|(left, right)| C::compress(&[left, right]))
                .collect_vec();
            digest_layers.push(next_digests);
        }
        Self {
            leaves,
            digest_layers,
        }
    }

    pub fn root(&self) -> D
    where
        D: Copy,
    {
        self.digest_layers.last().unwrap()[0]
    }
}

pub struct MerkleProof<T> {
    pub siblings: Vec<T>,
}

pub struct MerkleTreeVCS<L, D, H, C>
where
    H: CryptographicHasher<L, D>,
    C: CompressionFunction<D, 2>,
{
    _phantom_l: PhantomData<L>,
    _phantom_d: PhantomData<D>,
    _phantom_h: PhantomData<H>,
    _phantom_c: PhantomData<C>,
}

impl<L, D, H, C> Oracle<L> for MerkleTreeVCS<L, D, H, C>
where
    H: CryptographicHasher<L, D>,
    C: CompressionFunction<D, 2>,
{
    type ProverData = MerkleTree<L, D>;
    type Commitment = D;
    type Proof = MerkleProof<D>;
    type Error = ();

    fn open(_index: usize) -> (L, Self::Proof) {
        todo!()
    }

    fn verify(
        _commit: &D,
        _index: usize,
        _item: L,
        _proof: &MerkleProof<D>,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}

impl<L, D, H, C> ConcreteOracle<L> for MerkleTreeVCS<L, D, H, C>
where
    D: Copy,
    H: CryptographicHasher<L, D>,
    C: CompressionFunction<D, 2>,
{
    fn commit(input: Vec<L>) -> (Self::ProverData, Self::Commitment) {
        let tree = MerkleTree::new::<H, C>(input);
        let root = tree.root();
        (tree, root)
    }
}
