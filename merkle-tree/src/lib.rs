#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_commit::vector_commit::{ConcreteOracle, Oracle};
use p3_symmetric::hasher::CryptographicHasher;

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

impl<T, H, const OUT_WIDTH: usize> Oracle<T> for MerkleTreeVCS<T, H, OUT_WIDTH>
where
    H: CryptographicHasher<T, OUT_WIDTH>,
{
    type ProverData = MerkleTree<T>;
    type Commitment = [T; OUT_WIDTH];
    type Proof = MerkleProof<T>;
    type Error = ();

    fn open(_index: usize) -> (T, Self::Proof) {
        todo!()
    }

    fn verify(
        _commit: &[T; OUT_WIDTH],
        _index: usize,
        _item: T,
        _proof: &MerkleProof<T>,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}

impl<T, H, const OUT_WIDTH: usize> ConcreteOracle<T> for MerkleTreeVCS<T, H, OUT_WIDTH>
where
    H: CryptographicHasher<T, OUT_WIDTH>,
{
    fn commit(_input: Vec<T>) -> (Self::ProverData, Self::Commitment) {
        todo!()
    }
}
