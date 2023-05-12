use p3_air::{Air, AirBuilder};
use p3_baby_stark::{prove, StarkConfig};
use p3_fri::FRIBasedPCS;
use p3_lde::NaiveLDE;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMMCS;
use p3_mersenne_31::{Mersenne31, Mersenne31Complex};
use p3_poseidon::Poseidon;
use p3_symmetric::compression::TruncatedPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation, MDSPermutation};
use p3_symmetric::sponge::PaddingFreeSponge;
use rand::thread_rng;

struct MyConfig;

type F = Mersenne31;
struct MyMds;
impl CryptographicPermutation<[F; 8]> for MyMds {
    fn permute(&self, input: [F; 8]) -> [F; 8] {
        input // TODO
    }
}
impl ArrayPermutation<F, 8> for MyMds {}
impl MDSPermutation<F, 8> for MyMds {}

type MDS = MyMds;
type Perm = Poseidon<F, MDS, 8, 7>;
type H4 = PaddingFreeSponge<F, Perm, { 4 + 4 }>;
type C = TruncatedPermutation<F, Perm, 2, 4, { 2 * 4 }>;
type MMCS = MerkleTreeMMCS<F, [F; 4], H4, C>;
impl StarkConfig for MyConfig {
    type F = F;
    type Domain = Mersenne31Complex<F>;
    type Challenge = Self::F; // TODO: Use an extension.
    type PCS = FRIBasedPCS<Self::Challenge, MMCS, MMCS>;
    type LDE = NaiveLDE;
}

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
fn test_prove() {
    let mut rng = thread_rng();
    let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
    prove::<MyConfig, MulAir>(&MulAir, trace);
}
