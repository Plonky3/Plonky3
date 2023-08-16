use p3_air::{Air, AirBuilder};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2BowersFft;
use p3_fri::{FriBasedPcs, FriConfigImpl};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::compression::TruncatedPermutation;
use p3_symmetric::mds::MDSPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use p3_symmetric::sponge::PaddingFreeSponge;
use p3_uni_stark::{prove, StarkConfigImpl};
use rand::thread_rng;

struct MulAir;

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.row_slice(0);
        let diff = main_local[0] * main_local[1] - main_local[2];
        builder.assert_zero(diff);
    }
}

#[test]
#[ignore] // TODO: Not ready yet.
#[allow(clippy::items_after_statements)]
#[allow(clippy::upper_case_acronyms)]
fn test_prove_goldilocks() {
    type Val = Goldilocks;
    type Dom = Goldilocks;
    type Challenge = Goldilocks; // TODO

    #[derive(Clone)]
    struct MyMds;
    impl CryptographicPermutation<[Val; 8]> for MyMds {
        fn permute(&self, input: [Val; 8]) -> [Val; 8] {
            input // TODO
        }
    }
    impl ArrayPermutation<Val, 8> for MyMds {}
    impl MDSPermutation<Val, 8> for MyMds {}

    type MDS = MyMds;
    let mds = MyMds;

    type Perm = Poseidon<Val, MDS, 8, 7>;
    let perm = Perm::new(5, 5, vec![], mds);

    type H4 = PaddingFreeSponge<Val, Perm, { 4 + 4 }>;
    let h4 = H4::new(perm.clone());

    type C = TruncatedPermutation<Val, Perm, 2, 4, { 2 * 4 }>;
    let c = C::new(perm.clone());

    type Mmcs = MerkleTreeMmcs<Val, [Val; 4], H4, C>;
    type Dft = Radix2BowersFft;

    type Challenger = DuplexChallenger<Val, Perm, 8>;
    type MyFriConfig = FriConfigImpl<Val, Challenge, Mmcs, Mmcs, Challenger>;
    type Pcs = FriBasedPcs<MyFriConfig, Dft>;
    type MyConfig = StarkConfigImpl<Val, Dom, Challenge, Challenge, Pcs, Dft, Challenger>;

    let mut rng = thread_rng();
    let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
    let pcs = Pcs::new(Dft::default(), 1, Mmcs::new(h4, c));
    let config = StarkConfigImpl::new(pcs, Dft::default());
    let mut challenger = Challenger::new(perm);
    prove::<MyConfig, _, _>(&MulAir, &config, &mut challenger, trace);
}

#[test]
#[ignore] // TODO: Not ready yet.
fn test_prove_mersenne_31() {
    todo!()
}
