use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DitParallel;
use p3_fri::{FriBasedPcs, FriConfigImpl, FriLdt};
use p3_ldt::QuotientMmcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::compression::CompressionFunctionFromHasher;
use p3_symmetric::hasher::SerializingHasher32;
use p3_uni_stark::{prove, verify, StarkConfigImpl, VerificationError};
use rand::thread_rng;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// How many `a * b = c` operations to do per row in the AIR.
const REPETITIONS: usize = 10;
const TRACE_WIDTH: usize = REPETITIONS * 3;

struct MulAir;

impl<F> BaseAir<F> for MulAir {}

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.row_slice(0);

        for i in 0..REPETITIONS {
            let start = i * 3;
            let a = main_local[start];
            let b = main_local[start + 1];
            let c = main_local[start + 2];
            builder.assert_zero(a * b - c);
        }
    }
}

#[test]
fn test_prove_baby_bear() -> Result<(), VerificationError> {
    Registry::default()
        .with(EnvFilter::from_default_env())
        .with(ForestLayer::default())
        .init();

    const HEIGHT: usize = 1 << 6;

    type Val = BabyBear;
    type Domain = Val;
    type Challenge = Val; // TODO

    type MyMds = CosetMds<Val, 16>;
    let mds = MyMds::default();

    type Perm = Poseidon<Val, MyMds, 16, 5>;
    let perm = Perm::new_from_rng(4, 22, mds, &mut thread_rng()); // TODO: Use deterministic RNG

    type MyHash = SerializingHasher32<Val, Blake3>;
    let hash = MyHash::new(Blake3);

    type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 8>;
    let compress = MyCompress::new(hash);

    type MyMmcs = MerkleTreeMmcs<Val, [Val; 8], MyHash, MyCompress>;
    let mmcs = MyMmcs::new(hash, compress);

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16>;

    type Quotient = QuotientMmcs<Domain, Challenge, MyMmcs>;
    type MyFriConfig = FriConfigImpl<Val, Domain, Challenge, Quotient, MyMmcs, Challenger>;
    let fri_config = MyFriConfig::new(40, mmcs.clone());
    let ldt = FriLdt { config: fri_config };

    type Pcs = FriBasedPcs<MyFriConfig, MyMmcs, Dft, Challenger>;
    type MyConfig = StarkConfigImpl<Val, Domain, Challenge, Pcs, Dft, Challenger>;

    let mut rng = thread_rng();
    let trace = RowMajorMatrix::rand(&mut rng, HEIGHT, TRACE_WIDTH);
    let pcs = Pcs::new(dft, 1, mmcs, ldt);
    let config = StarkConfigImpl::new(pcs, Dft {});
    let mut challenger = Challenger::new(perm.clone());
    let proof = prove::<MyConfig, _>(&config, &MulAir, &mut challenger, trace);

    let mut challenger = Challenger::new(perm);
    verify(&config, &MulAir, &mut challenger, &proof)
}

#[test]
#[ignore] // TODO: Not ready yet.
fn test_prove_mersenne_31() {
    todo!()
}
