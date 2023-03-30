#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_symmetric::hasher::CryptographicHasher;
use p3_symmetric::vector_commit::VectorCommitmentScheme;

pub struct MerkleTree<T> {
    pub leaves: Vec<Vec<T>>,
}

pub struct MerkleProof<T> {
    pub siblings: Vec<T>,
}

pub struct MerkleTreeVCS<T, H, const OUT_WIDTH: usize>
where
    H: CryptographicHasher<T, OUT_WIDTH>,
{
    _phantom_t: PhantomData<T>,
    _phantom_h: PhantomData<H>,
}

impl<T, H, const OUT_WIDTH: usize> VectorCommitmentScheme for MerkleTreeVCS<T, H, OUT_WIDTH>
where
    H: CryptographicHasher<T, OUT_WIDTH>,
{
    type Item = T;
    type ProverData = MerkleTree<T>;
    type Commitment = [T; OUT_WIDTH];
    type Proof = MerkleProof<T>;
    type Error = ();

    fn commit(_input: Vec<Self::Item>) -> (Self::ProverData, Self::Commitment) {
        todo!()
    }

    fn open(_index: usize) -> (Self::Item, Self::Proof) {
        todo!()
    }

    fn verify(
        _commit: Self::Commitment,
        _index: usize,
        _item: Self::Item,
        _proof: Self::Proof,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}
