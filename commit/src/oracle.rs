use alloc::vec::Vec;

/// A prover-supplied oracle.
///
/// This is essentially a (not necessarily hiding) vector commitment to `T`. However, this API
/// supports virtual oracles as well as concrete ones (`ConcreteOracle`).
pub trait Oracle<T> {
    type ProverData;
    type Commitment;
    type Proof;
    type Error;

    fn open(index: usize, prover_data: &Self::ProverData) -> (T, Self::Proof);

    fn verify(
        commit: &Self::Commitment,
        index: usize,
        item: T,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

pub trait ConcreteOracle<T>: Oracle<T> {
    fn commit(input: Vec<T>) -> (Self::ProverData, Self::Commitment);
}

// TODO: Streaming oracle? Where `ProverData` can be initialized and then incrementally updated,
// rather than being created all at once.
