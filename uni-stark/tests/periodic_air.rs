use core::fmt::Debug;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, AirWithPeriodicColumns, BaseAir, PeriodicAirBuilder};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::testing::TrivialPcs;
use p3_commit::{EvaluatePolynomialAtPoint, ExtensionMmcs};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{Domain, StarkConfig, StarkGenericConfig, Val, prove, verify};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

const WIDTH: usize = 2;

pub struct PeriodicAir<F> {
    periodic: Vec<Vec<F>>,
}

impl<F: Field> Default for PeriodicAir<F> {
    fn default() -> Self {
        Self {
            periodic: vec![
                vec![
                    F::from_usize(1),
                    F::from_usize(2),
                    F::from_usize(3),
                    F::from_usize(4),
                ],
                vec![F::from_usize(10), F::from_usize(20)],
            ],
        }
    }
}

impl<F: Field> PeriodicAir<F> {
    pub fn valid_trace(&self, rows: usize) -> RowMajorMatrix<F> {
        let periodic = AirWithPeriodicColumns::periodic_columns(self);
        let mut values = F::zero_vec(rows * WIDTH);
        for (i, row) in values.chunks_exact_mut(WIDTH).enumerate() {
            row[0] = periodic[0][i % periodic[0].len()];
            row[1] = periodic[1][i % periodic[1].len()];
        }
        RowMajorMatrix::new(values, WIDTH)
    }
}

impl<F: Field> BaseAir<F> for PeriodicAir<F> {
    fn width(&self) -> usize {
        WIDTH
    }

    fn num_periodic_columns(&self) -> usize {
        self.periodic.len()
    }

    fn periodic_columns(&self) -> &[Vec<F>] {
        &self.periodic
    }
}

impl<F: Field> AirWithPeriodicColumns<F> for PeriodicAir<F> {}

impl<AB: AirBuilder + PeriodicAirBuilder> Air<AB> for PeriodicAir<AB::F>
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("matrix has rows");
        let p0 = builder.periodic_values()[0].into();
        let p1 = builder.periodic_values()[1].into();
        builder.assert_eq(local[0].clone(), p0);
        builder.assert_eq(local[1].clone(), p1);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn do_test<SC: StarkGenericConfig>(
    config: SC,
    air: PeriodicAir<Val<SC>>,
    rows: usize,
) -> Result<(), impl Debug>
where
    SC::Challenger: Clone,
    StandardUniform: Distribution<Val<SC>>,
    Domain<SC>: EvaluatePolynomialAtPoint,
{
    let trace = air.valid_trace(rows);

    let proof = prove(&config, &air, trace, &[]);

    let serialized_proof = postcard::to_allocvec(&proof).expect("unable to serialize proof");
    let deserialized_proof =
        postcard::from_bytes(&serialized_proof).expect("unable to deserialize proof");

    verify(&config, &air, &deserialized_proof, &[])
}

fn do_test_bb_trivial(log_n: usize) -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    type Pcs = TrivialPcs<Val, Radix2DitParallel<Val>>;
    let pcs = TrivialPcs {
        dft,
        log_n,
        _phantom: PhantomData,
    };
    let challenger = Challenger::new(perm);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = PeriodicAir::<Val>::default();

    do_test(config, air, 1 << log_n)
}

fn do_test_bb_twoadic(log_blowup: usize, log_n: usize) -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 3,
        max_log_arity: 2,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = PeriodicAir::<Val>::default();

    do_test(config, air, 1 << log_n)
}

fn do_test_m31_circle(log_blowup: usize, log_n: usize) -> Result<(), impl Debug> {
    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: PhantomData,
    };
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = PeriodicAir::<Val>::default();

    do_test(config, air, 1 << log_n)
}

#[test]
fn prove_bb_trivial() -> Result<(), impl Debug> {
    do_test_bb_trivial(8)
}

#[test]
fn prove_bb_twoadic() -> Result<(), impl Debug> {
    do_test_bb_twoadic(1, 6)
}

#[test]
fn prove_m31_circle() -> Result<(), impl Debug> {
    do_test_m31_circle(1, 6)
}
