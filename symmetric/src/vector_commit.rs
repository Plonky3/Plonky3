use alloc::vec::Vec;

pub trait VectorCommitmentScheme {
    type Item;
    type ProverData;
    type Commitment;
    type Proof;
    type Error;

    fn commit(input: Vec<Self::Item>) -> (Self::ProverData, Self::Commitment);
    fn open(index: usize) -> (Self::Item, Self::Proof);
    fn verify(
        commit: Self::Commitment,
        index: usize,
        item: Self::Item,
        proof: Self::Proof,
    ) -> Result<(), Self::Error>;
}
