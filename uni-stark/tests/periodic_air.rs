use core::fmt::Debug;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[derive(Clone)]
struct PeriodicAir<F> {
    periodic: Vec<Vec<F>>,
}

impl<F: Field + PrimeCharacteristicRing> PeriodicAir<F> {
    fn new() -> Self {
        Self {
            periodic: vec![
                vec![
                    F::from_u64(1),
                    F::from_u64(2),
                    F::from_u64(3),
                    F::from_u64(4),
                ],
                vec![F::from_u64(10), F::from_u64(20)],
            ],
        }
    }

    fn valid_trace(&self, rows: usize) -> RowMajorMatrix<F> {
        let mut values = F::zero_vec(rows * 2);
        for (i, row) in values.chunks_exact_mut(2).enumerate() {
            row[0] = self.periodic[0][i % self.periodic[0].len()];
            row[1] = self.periodic[1][i % self.periodic[1].len()];
        }
        RowMajorMatrix::new(values, 2)
    }
}

impl<F: Field> BaseAir<F> for PeriodicAir<F> {
    fn width(&self) -> usize {
        2
    }

    fn num_periodic_columns(&self) -> usize {
        self.periodic.len()
    }

    fn periodic_columns(&self) -> Vec<Vec<F>> {
        self.periodic.clone()
    }
}

impl<AB: AirBuilder> Air<AB> for PeriodicAir<AB::F>
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let p0 = builder.periodic_values()[0].into();
        let p1 = builder.periodic_values()[1].into();
        builder.assert_eq(local[0], p0);
        builder.assert_eq(local[1], p1);
    }
}

#[test]
fn periodic_air_two_adic_prove_verify() -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2DitParallel<Val>;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 3,
        max_log_arity: 2,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    let config = Config::new(pcs, challenger);

    let air = PeriodicAir::<Val>::new();
    let trace = air.valid_trace(1 << 6);
    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[])
}

#[test]
fn periodic_air_two_adic_zk_prove_verify() -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = MerkleTreeHidingMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        Hash,
        Compress,
        SmallRng,
        2,
        8,
        4,
    >;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2DitParallel<Val>;
    type Pcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, SmallRng>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0, rng);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing_zk(challenge_mmcs);
    let pcs = Pcs::new(dft, val_mmcs, fri_params, 4, SmallRng::seed_from_u64(2));
    let challenger = Challenger::new(perm);
    let config = Config::new(pcs, challenger);

    let air = PeriodicAir::<Val>::new();
    let trace = air.valid_trace(1 << 6);
    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[])
}

#[test]
fn periodic_air_circle_prove_verify() -> Result<(), impl Debug> {
    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>;
    type ByteHash = p3_keccak::Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 2, 32>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = p3_challenger::SerializingChallenger32<
        Val,
        p3_challenger::HashChallenger<u8, ByteHash, 32>,
    >;
    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = Compress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: PhantomData,
    };
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    let config = Config::new(pcs, challenger);

    let air = PeriodicAir::<Val>::new();
    let trace = air.valid_trace(1 << 5);
    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[])
}
