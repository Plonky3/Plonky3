use alloc::vec::Vec;
use p3_matrix::dense::DenseMatrix;

/// An oracle which holds a batch of vectors, potentially of different sizes, and supports querying
/// all vectors at a particular index.
///
/// These oracles may be concrete (`ConcreteOracle`) or virtual.
// TODO: Rename? MMCS = Mixed Matrix Commitment Scheme?
pub trait Oracle<T> {
    type ProverData;
    type Commitment;
    type Proof;
    type Error;

    fn open_batch(row: usize, prover_data: &Self::ProverData) -> (Vec<&[T]>, Self::Proof);

    /// Verify a batch opening.
    fn verify_batch(
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        item: Vec<T>,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

pub struct Dimensions {
    pub width: usize,
    pub log2_height: usize,
}

// pub trait TwoAdicOracle<T>: Oracle<T> {
//     /// The log2 of the size of the `i`th oracle.
//     fn log2_size(i: usize) -> usize;
// }

pub trait ConcreteOracle<T>: Oracle<T> {
    fn commit_batch(inputs: Vec<DenseMatrix<T>>) -> (Self::ProverData, Self::Commitment);
}

// TODO: Streaming oracle? Where `ProverData` can be initialized and then incrementally updated,
// rather than being created all at once.
