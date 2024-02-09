// use p3_air::{Air, AirBuilder};
// use p3_multi_stark::{prove, StarkConfig};
// use p3_challenger::DuplexChallenger;
// use p3_fri::FRIBasedPcs;
// use p3_lde::NaiveCosetLde;
// use p3_matrix::dense::RowMajorMatrix;
// use p3_matrix::Matrix;
// use p3_merkle_tree::MerkleTreeMMCS;
// use p3_poseidon::Poseidon;
// use p3_symmetric::TruncatedPermutation;
// use p3_symmetric::{ArrayPermutation, CryptographicPermutation, MDSPermutation};
// use p3_symmetric::PaddingFreeSponge;
// use rand::thread_rng;
// use p3_mersenne_31::Mersenne31;
// use p3_tensor_pcs::TensorPcs;
//
// struct MulAir;
//
// impl<AB: AirBuilder> Air<AB> for MulAir {
//     fn eval(&self, builder: &mut AB) {
//         let main = builder.main();
//         let main_local = main.row(0);
//         let diff = main_local[0] * main_local[1] - main_local[2];
//         builder.assert_zero(diff);
//     }
// }
//
// #[test]
// #[ignore] // TODO: Not ready yet.
// fn test_prove_goldilocks() {
//     type Val = Mersenne31;
//     type Challenge = Mersenne31; // TODO
//
//     #[derive(Clone)]
//     struct MyMds;
//     impl CryptographicPermutation<[Val; 8]> for MyMds {
//         fn permute(&self, input: [Val; 8]) -> [Val; 8] {
//             input // TODO
//         }
//     }
//     impl ArrayPermutation<Val, 8> for MyMds {}
//     impl MdsPermutation<Val, 8> for MyMds {}
//
//     type Mds = MyMds;
//     let mds = MyMds;
//
//     type Perm = Poseidon<Val, Mds, 8, 7>;
//     let perm = Perm::new(5, 5, vec![], mds);
//
//     type H4 = PaddingFreeSponge<Val, Perm, { 4 + 4 }>;
//     let h4 = H4::new(perm.clone());
//
//     type C = TruncatedPermutation<Val, Perm, 2, 4, { 2 * 4 }>;
//     let c = C::new(perm.clone());
//
//     type Mmcs = MerkleTreeMMCS<Val, [Val; 4], H4, C>;
//     type Pcs = TensorPcs<Val, MyCode>;
//     type MyConfig = StarkConfig<Val, Challenge, Challenge, Pcs>;
//
//     let mut rng = thread_rng();
//     let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
//     let pcs = todo!();
//     let config = StarkConfig::new(pcs);
//     let mut challenger = DuplexChallenger::new(perm);
//     prove::<MyConfig, _, _>(&MulAir, config, &mut challenger, trace);
// }
//
// #[test]
// #[ignore] // TODO: Not ready yet.
// fn test_prove_mersenne_31() {
//     todo!()
// }
