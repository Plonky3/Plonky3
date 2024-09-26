use std::fmt::Debug;
use std::marker::PhantomData;

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::testing::TrivialPcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher32, TruncatedPermutation,
};
use p3_uni_stark::{prove, verify, StarkConfig, StarkGenericConfig, Val};
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

/// How many `a * b = c` operations to do per row in the AIR.
const REPETITIONS: usize = 20;
const TRACE_WIDTH: usize = REPETITIONS * 3;

/*
In its basic form, asserts a^(self.degree-1) * b = c
(so that the total constraint degree is self.degree)


If `uses_transition_constraints`, checks that on transition rows, the first a = row number
*/
pub struct MulAir {
    degree: u64,
    uses_boundary_constraints: bool,
    uses_transition_constraints: bool,
}

impl Default for MulAir {
    fn default() -> Self {
        MulAir {
            degree: 3,
            uses_boundary_constraints: true,
            uses_transition_constraints: true,
        }
    }
}

impl MulAir {
    pub fn random_valid_trace<F: Field>(&self, rows: usize, valid: bool) -> RowMajorMatrix<F>
    where
        Standard: Distribution<F>,
    {
        let mut rng = thread_rng();
        let mut trace_values = F::zero_vec(rows * TRACE_WIDTH);
        for (i, (a, b, c)) in trace_values.iter_mut().tuples().enumerate() {
            let row = i / REPETITIONS;

            *a = if self.uses_transition_constraints {
                F::from_canonical_usize(i)
            } else {
                rng.gen()
            };
            *b = if self.uses_boundary_constraints && row == 0 {
                a.square() + F::one()
            } else {
                rng.gen()
            };
            *c = a.exp_u64(self.degree - 1) * *b;

            if !valid {
                // make it invalid
                *c *= F::two();
            }
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
        let main_next = main.row_slice(1);

        for i in 0..REPETITIONS {
            let start = i * 3;
            let a = main_local[start];
            let b = main_local[start + 1];
            let c = main_local[start + 2];
            builder.assert_zero(a.into().exp_u64(self.degree - 1) * b - c);
            if self.uses_boundary_constraints {
                builder
                    .when_first_row()
                    .assert_eq(a * a + AB::Expr::one(), b);
            }
            if self.uses_transition_constraints {
                let next_a = main_next[start];
                builder
                    .when_transition()
                    .assert_eq(a + AB::Expr::from_canonical_usize(REPETITIONS), next_a);
            }
        }
    }
}

fn do_test<SC: StarkGenericConfig>(
    config: SC,
    air: MulAir,
    log_height: usize,
    challenger: SC::Challenger,
) -> Result<(), impl Debug>
where
    SC::Challenger: Clone,
    Standard: Distribution<Val<SC>>,
{
    let trace = air.random_valid_trace(log_height, true);

    let mut p_challenger = challenger.clone();
    let proof = prove(&config, &air, &mut p_challenger, trace, &vec![]);

    let serialized_proof = postcard::to_allocvec(&proof).expect("unable to serialize proof");
    tracing::debug!("serialized_proof len: {} bytes", serialized_proof.len());

    let deserialized_proof =
        postcard::from_bytes(&serialized_proof).expect("unable to deserialize proof");

    let mut v_challenger = challenger.clone();
    verify(
        &config,
        &air,
        &mut v_challenger,
        &deserialized_proof,
        &vec![],
    )
}

fn do_test_bb_trivial(degree: u64, log_n: usize) -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    type Pcs = TrivialPcs<Val, Radix2DitParallel>;
    let pcs = TrivialPcs {
        dft,
        log_n,
        _phantom: PhantomData,
    };

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let air = MulAir {
        degree,
        ..Default::default()
    };

    do_test(config, air, 1 << log_n, Challenger::new(perm))
}

#[test]
fn prove_bb_trivial_deg2() -> Result<(), impl Debug> {
    do_test_bb_trivial(2, 8)
}

#[test]
fn prove_bb_trivial_deg3() -> Result<(), impl Debug> {
    do_test_bb_trivial(3, 8)
}

#[test]
fn prove_bb_trivial_deg4() -> Result<(), impl Debug> {
    do_test_bb_trivial(4, 8)
}

fn do_test_bb_twoadic(log_blowup: usize, degree: u64, log_n: usize) -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    let fri_config = FriConfig {
        log_blowup,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let air = MulAir {
        degree,
        ..Default::default()
    };

    do_test(config, air, 1 << log_n, Challenger::new(perm))
}

#[test]
fn prove_bb_twoadic_deg2() -> Result<(), impl Debug> {
    do_test_bb_twoadic(1, 2, 7)
}

#[test]
fn prove_bb_twoadic_deg3() -> Result<(), impl Debug> {
    do_test_bb_twoadic(1, 3, 7)
}

#[test]
fn prove_bb_twoadic_deg4() -> Result<(), impl Debug> {
    do_test_bb_twoadic(2, 4, 6)
}

#[test]
fn prove_bb_twoadic_deg5() -> Result<(), impl Debug> {
    do_test_bb_twoadic(2, 5, 6)
}

fn do_test_m31_circle(log_blowup: usize, degree: u64, log_n: usize) -> Result<(), impl Debug> {
    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher32<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let fri_config = FriConfig {
        log_blowup,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_config,
        _phantom: PhantomData,
    };

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let air = MulAir {
        degree,
        uses_boundary_constraints: true,
        uses_transition_constraints: true,
    };

    do_test(
        config,
        air,
        1 << log_n,
        Challenger::from_hasher(vec![], byte_hash),
    )
}

#[test]
fn prove_m31_circle_deg2() -> Result<(), impl Debug> {
    do_test_m31_circle(1, 2, 8)
}

#[test]
fn prove_m31_circle_deg3() -> Result<(), impl Debug> {
    do_test_m31_circle(1, 3, 9)
}
