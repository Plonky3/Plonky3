#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_commit::oracle::{ConcreteOracle, Oracle};
use p3_symmetric::compression::CompressionFunction;
use p3_symmetric::hasher::CryptographicHasher;

// TODO: Add a variant that supports compression?
// How would we keep track of previously-seen paths - make the Oracle methods take &mut self?

/// A standard binary Merkle tree.
pub struct MerkleTree<T> {
    pub leaves: Vec<Vec<T>>,
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
    type ProverData = MerkleTree<L>;
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
    H: CryptographicHasher<L, D>,
    C: CompressionFunction<D, 2>,
{
    fn commit(_input: Vec<L>) -> (Self::ProverData, Self::Commitment) {
        todo!()
    }
}
