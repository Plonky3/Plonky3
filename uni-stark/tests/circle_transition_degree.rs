//! Regression test for the circle transition-constraint degree accounting.
//!
//! When the maximum-degree constraint of an AIR is a *transition* constraint, the circle
//! transition selector must not inflate the quotient degree beyond what the quotient-chunk
//! count provisions (eprint 2024/278, Remark 17; upstream issue #575). With the degree-N/2
//! selector this over-counts and the proof fails to verify; with the tangent functional the
//! transition constraint carries the full AIR degree with no selector penalty.

use core::fmt::Debug;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// AIR with a single transition constraint of total degree `degree`:
/// on every transition row, `next_b = a^(degree - 1) * b`. Column `a` is free.
struct TransitionMaxDegAir {
    degree: u64,
}

impl<F> BaseAir<F> for TransitionMaxDegAir {
    fn width(&self) -> usize {
        2
    }
}

impl<AB: AirBuilder> Air<AB> for TransitionMaxDegAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        let a = local[0];
        let b = local[1];
        let next_b = next[1];
        builder
            .when_transition()
            .assert_eq(a.into().exp_u64(self.degree - 1) * b, next_b);
    }
}

impl TransitionMaxDegAir {
    fn valid_trace<F: Field>(&self, rows: usize) -> RowMajorMatrix<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let a: Vec<F> = (0..rows).map(|_| rng.random()).collect();
        let mut b = F::zero_vec(rows);
        b[0] = rng.random();
        for i in 1..rows {
            b[i] = a[i - 1].exp_u64(self.degree - 1) * b[i - 1];
        }
        let mut values = F::zero_vec(rows * 2);
        for i in 0..rows {
            values[2 * i] = a[i];
            values[2 * i + 1] = b[i];
        }
        RowMajorMatrix::new(values, 2)
    }
}

fn prove_circle_transition(degree: u64, log_n: usize) -> Result<(), impl Debug> {
    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress, 0);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let fri_params = FriParameters {
        log_blowup: 1,
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

    let air = TransitionMaxDegAir { degree };
    let trace = air.valid_trace::<Val>(1 << log_n);

    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[])
}

#[test]
fn prove_circle_transition_deg2() -> Result<(), impl Debug> {
    prove_circle_transition(2, 6)
}

#[test]
fn prove_circle_transition_deg3() -> Result<(), impl Debug> {
    prove_circle_transition(3, 7)
}

#[test]
fn prove_circle_transition_deg4() -> Result<(), impl Debug> {
    prove_circle_transition(4, 7)
}

#[test]
fn prove_circle_transition_deg5() -> Result<(), impl Debug> {
    prove_circle_transition(5, 7)
}
