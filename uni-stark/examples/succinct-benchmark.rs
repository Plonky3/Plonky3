use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriBasedPcs, FriConfigImpl, FriLdt};
use p3_ldt::QuotientMmcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfigImpl};
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

//  2^26 rows and 300 columns
/// How many `a * b = c` operations to do per row in the AIR.
const REPETITIONS: usize = 300;
const TRACE_WIDTH: usize = REPETITIONS * 3;

struct MulAir;

impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        TRACE_WIDTH
    }
}

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

            // TODO: Temporarily added this silly degree 3 constraint because we're getting an
            // OodEvaluationMismatch when log_quotient_degree = 0.
            builder.assert_zero(a * b * c - c * b * a);
        }
    }
}

fn random_valid_trace<F: Field>(rows: usize) -> RowMajorMatrix<F>
where
    Standard: Distribution<F>,
{
    let mut rng = thread_rng();
    let mut trace_values = vec![F::default(); rows * TRACE_WIDTH];
    for (a, b, c) in trace_values.iter_mut().tuples() {
        *a = rng.gen();
        *b = rng.gen();
        *c = *a * *b;
    }
    RowMajorMatrix::new(trace_values, TRACE_WIDTH)
}

fn main () {
    let env_filter = EnvFilter::builder()
    .with_default_directive(LevelFilter::INFO.into())
    .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    const HEIGHT: usize = 1 << 3;

    type Val = BabyBear;
    type Domain = Val;
    type Challenge = BinomialExtensionField<Val, 4>;
    type PackedChallenge = BinomialExtensionField<<Domain as Field>::Packing, 4>;

    type MyMds = CosetMds<Val, 16>;
    let mds = MyMds::default();

    type Perm = Poseidon2<Val, MyMds, DiffusionMatrixBabybear, 16, 5>;
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<<Val as Field>::Packing, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16>;

    type Quotient = QuotientMmcs<Domain, Challenge, ValMmcs>;
    type MyFriConfig = FriConfigImpl<Val, Challenge, Quotient, ChallengeMmcs, Challenger>;
    let fri_config = MyFriConfig::new(40, challenge_mmcs);
    let ldt = FriLdt { config: fri_config };

    type Pcs = FriBasedPcs<MyFriConfig, ValMmcs, Dft, Challenger>;
    type MyConfig = StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>;

    let pcs = Pcs::new(dft, val_mmcs, ldt);
    let config = StarkConfigImpl::new(pcs);
    let mut challenger = Challenger::new(perm.clone());
    let trace = random_valid_trace::<Val>(HEIGHT);
    let proof = prove::<MyConfig, _>(&config, &MulAir, &mut challenger, trace);

    let mut challenger = Challenger::new(perm);
    let _ = verify(&config, &MulAir, &mut challenger, &proof);
}
