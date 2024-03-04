use std::marker::PhantomData;

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabybear};
use p3_challenger::DuplexChallenger;
use p3_commit::testing::TrivialPcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::Poseidon2;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig, StarkGenericConfig, Val, VerificationError};
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

/// How many `a * b = c` operations to do per row in the AIR.
const REPETITIONS: usize = 10;
const TRACE_WIDTH: usize = REPETITIONS * 3;

pub struct MulAir {
    degree: u64,
}

impl MulAir {
    pub fn random_valid_trace<F: Field>(&self, rows: usize, valid: bool) -> RowMajorMatrix<F>
    where
        Standard: Distribution<F>,
    {
        let mut rng = thread_rng();
        let mut trace_values = vec![F::default(); rows * TRACE_WIDTH];
        for (a, b, c) in trace_values.iter_mut().tuples() {
            *a = rng.gen();
            *b = rng.gen();
            *c = a.exp_u64(self.degree - 1) * *b + if valid { F::zero() } else { F::one() };
        }
        RowMajorMatrix::new(trace_values, TRACE_WIDTH)
    }
}

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
            let a = main_local[start].into().exp_u64(self.degree - 1);
            let b = main_local[start + 1];
            let c = main_local[start + 2];
            builder.assert_zero(a * b - c);
        }
    }
}

fn do_test<SC: StarkGenericConfig>(
    config: SC,
    degree: u64,
    log_height: usize,
    challenger: SC::Challenger,
) -> Result<(), VerificationError>
where
    SC::Challenger: Clone,
    Standard: Distribution<Val<SC>>,
{
    let air = MulAir { degree };
    let trace = air.random_valid_trace(log_height, true);

    let mut p_challenger = challenger.clone();
    let proof = prove(&config, &air, &mut p_challenger, trace);

    let serialized_proof = postcard::to_allocvec(&proof).expect("unable to serialize proof");
    tracing::debug!("serialized_proof len: {} bytes", serialized_proof.len());

    let deserialized_proof =
        postcard::from_bytes(&serialized_proof).expect("unable to deserialize proof");

    let mut v_challenger = challenger.clone();
    verify(&config, &air, &mut v_challenger, &deserialized_proof)
}

fn do_test_bb_trivial(degree: u64, log_n: usize) -> Result<(), VerificationError> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, DiffusionMatrixBabybear, 16, 7>;
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16>;

    type Pcs = TrivialPcs<Val, Radix2DitParallel>;
    let pcs = p3_commit::testing::TrivialPcs {
        dft,
        log_n,
        _phantom: PhantomData,
    };

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    do_test(config, degree, 1 << log_n, Challenger::new(perm))
}

#[test]
fn prove_bb_trivial_deg2() -> Result<(), VerificationError> {
    do_test_bb_trivial(2, 10)
}

#[test]
fn prove_bb_trivial_deg3() -> Result<(), VerificationError> {
    do_test_bb_trivial(3, 10)
}

#[test]
fn prove_bb_trivial_deg4() -> Result<(), VerificationError> {
    do_test_bb_trivial(4, 10)
}

fn do_test_bb_twoadic(
    log_blowup: usize,
    degree: u64,
    log_n: usize,
) -> Result<(), VerificationError> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, DiffusionMatrixBabybear, 16, 7>;
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16>;

    let fri_config = FriConfig {
        log_blowup,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(log_n, dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    do_test(config, degree, 1 << log_n, Challenger::new(perm))
}

#[test]
fn prove_bb_twoadic_deg2() -> Result<(), VerificationError> {
    do_test_bb_twoadic(1, 2, 10)
}

#[test]
fn prove_bb_twoadic_deg3() -> Result<(), VerificationError> {
    do_test_bb_twoadic(1, 3, 10)
}

#[test]
fn prove_bb_twoadic_deg4() -> Result<(), VerificationError> {
    do_test_bb_twoadic(2, 4, 10)
}

#[test]
fn prove_bb_twoadic_deg5() -> Result<(), VerificationError> {
    do_test_bb_twoadic(2, 5, 10)
}
