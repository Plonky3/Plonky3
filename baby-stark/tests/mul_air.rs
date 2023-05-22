use p3_air::{Air, AirBuilder};
use p3_baby_stark::{prove, StarkConfigImpl};
use p3_fri::FRIBasedPCS;
use p3_goldilocks::Goldilocks;
use p3_lde::NaiveCosetLDE;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMMCS;
use p3_poseidon::Poseidon;
use p3_symmetric::compression::TruncatedPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation, MDSPermutation};
use p3_symmetric::sponge::PaddingFreeSponge;
use rand::thread_rng;

struct MulAir;

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.row(0);
        let diff = main_local[0] * main_local[1] - main_local[2];
        builder.assert_zero(diff);
    }
}

#[test]
#[ignore] // TODO: Not ready yet.
fn test_prove_goldilocks() {
    type Val = Goldilocks;
    type Domain = Goldilocks;
    type Challenge = Goldilocks; // TODO
    struct MyMds;
    impl CryptographicPermutation<[Val; 8]> for MyMds {
        fn permute(&self, input: [Val; 8]) -> [Val; 8] {
            input // TODO
        }
    }
    impl ArrayPermutation<Val, 8> for MyMds {}
    impl MDSPermutation<Val, 8> for MyMds {}

    type MDS = MyMds;
    type Perm = Poseidon<Val, MDS, 8, 7>;
    type H4 = PaddingFreeSponge<Val, Perm, { 4 + 4 }>;
    type C = TruncatedPermutation<Val, Perm, 2, 4, { 2 * 4 }>;
    type MMCS = MerkleTreeMMCS<Val, [Val; 4], H4, C>;
    type LDE = NaiveCosetLDE;
    type PCS = FRIBasedPCS<Val, Domain, Challenge, LDE, MMCS, MMCS>;
    type MyConfig = StarkConfigImpl<Val, Domain, Challenge, PCS, LDE>;

    let mut rng = thread_rng();
    let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
    let pcs = PCS::new(NaiveCosetLDE);
    let config = StarkConfigImpl::new(pcs, NaiveCosetLDE);
    prove::<MyConfig, MulAir>(&MulAir, config, trace);
}

#[test]
#[ignore] // TODO: Not ready yet.
fn test_prove_mersenne_31() {
    todo!()
}
