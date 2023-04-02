use alloc::vec::Vec;

/// A vector commitment scheme (VCS).
pub trait VectorCommitmentScheme<T> {
    type ProverData;
    type Commitment;
    type Proof;
    type Error;

    fn commit(input: Vec<T>) -> (Self::ProverData, Self::Commitment);

    fn open(index: usize) -> (T, Self::Proof);

    fn verify(
        commit: Self::Commitment,
        index: usize,
        item: T,
        proof: Self::Proof,
    ) -> Result<(), Self::Error>;
}
