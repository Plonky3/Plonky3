use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_merkle_tree::MerkleTree;

pub mod reader;
pub mod writer;

pub type ProverDataExt<F, EF, const N: usize, const DIGEST_ELEMS: usize> =
    MerkleTree<F, F, FlatMatrixView<F, EF, DenseMatrix<EF>>, N, DIGEST_ELEMS>;

pub type ProverData<F, const N: usize, const DIGEST_ELEMS: usize> =
    MerkleTree<F, F, DenseMatrix<F>, N, DIGEST_ELEMS>;
