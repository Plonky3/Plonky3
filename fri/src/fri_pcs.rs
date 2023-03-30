// use alloc::vec::Vec;
// use hyperfield::field::Field;
// use hyperfield::matrix::dense::DenseMatrix;
// use hyperpcs::{PCS, UnivariatePCS};
// use crate::proof::FriProof;
//
// pub struct FriPcs<F: Field, H: H>;
//
// impl<F: Field> PCS<F> for FriPcs {
//     type Commitment = Hash;
//     type ProverData = ProverData;
//     type Proof = FriProof<F>;
//
//     fn commit_batches(polynomials: Vec<DenseMatrix<F>>) -> (Commitment, ProverData) {
//         todo!()
//     }
// }
//
// impl<F: Field> UnivariatePCS<F> for FriPcs {}
//
// struct ProverData {
//     // merkle_tree: MerkleTree<F, Hasher>,
// }
