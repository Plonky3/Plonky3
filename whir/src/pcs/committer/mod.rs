use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;

pub mod reader;
pub mod writer;

pub type ProverData<F, const N: usize, const DIGEST_ELEMS: usize> =
    MerkleTree<F, F, DenseMatrix<F>, N, DIGEST_ELEMS>;
