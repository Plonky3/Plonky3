//! Regression test for the Remark 22 quotient extension together with preprocessed columns.
//!
//! A maximum-degree *transition* constraint whose degree puts `d - 1` on a power-of-two boundary
//! triggers the circle extension chunk (eprint 2024/278, Remark 22). When such a constraint also
//! reads a preprocessed column, the extension chunk must evaluate that column on the small
//! extension domain, or the reconstructed quotient omits its contribution and verification fails.

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
use p3_uni_stark::{
    StarkConfig, prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed,
};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Nonzero, non-constant preprocessed coefficient for row `i`. Kept in one place so the AIR's
/// preprocessed trace and the witness generation agree.
fn coeff<F: PrimeCharacteristicRing>(i: usize) -> F {
    F::from_u64((i % 3 + 1) as u64)
}

/// Degree-3 transition AIR: on every transition row, `next_b = coeff * a * b`, where `coeff` is a
/// preprocessed column. `d - 1 = 2` is a power of two, so the Remark 22 extension chunk is
/// exercised; `coeff` participates multiplicatively, so an extension chunk that dropped it would
/// reconstruct the wrong quotient.
struct PreprocessedTransitionAir {
    rows: usize,
}

impl<F: Field> BaseAir<F> for PreprocessedTransitionAir {
    fn width(&self) -> usize {
        2
    }
    fn preprocessed_width(&self) -> usize {
        1
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(RowMajorMatrix::new(
            (0..self.rows).map(coeff::<F>).collect(),
            1,
        ))
    }
}

impl<AB: AirBuilder> Air<AB> for PreprocessedTransitionAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let (a, b, next_b) = {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();
            (local[0], local[1], next[1])
        };
        let coeff = {
            let preprocessed = builder.preprocessed();
            preprocessed.current_slice()[0]
        };
        builder.when_transition().assert_eq(coeff * a * b, next_b);
    }
}

impl PreprocessedTransitionAir {
    fn valid_trace<F: Field>(&self) -> RowMajorMatrix<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let a: Vec<F> = (0..self.rows).map(|_| rng.random()).collect();
        let mut b = F::zero_vec(self.rows);
        b[0] = rng.random();
        for i in 1..self.rows {
            b[i] = coeff::<F>(i - 1) * a[i - 1] * b[i - 1];
        }
        let mut values = F::zero_vec(self.rows * 2);
        for i in 0..self.rows {
            values[2 * i] = a[i];
            values[2 * i + 1] = b[i];
        }
        RowMajorMatrix::new(values, 2)
    }
}

type Val = Mersenne31;
type Challenge = BinomialExtensionField<Val, 3>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn make_config() -> (MyConfig, ByteHash) {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
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
    (MyConfig::new(pcs, challenger), byte_hash)
}

#[test]
fn preprocessed_transition_hits_extension_chunk() -> Result<(), impl Debug> {
    let log_n = 7;
    let (config, _) = make_config();
    let air = PreprocessedTransitionAir { rows: 1 << log_n };
    let trace = air.valid_trace::<Val>();
    let (pp_prover, pp_vk) =
        setup_preprocessed::<MyConfig, _>(&config, &air, log_n).expect("preprocessed setup");
    let proof = prove_with_preprocessed(&config, &air, trace, &[], Some(&pp_prover));
    verify_with_preprocessed(&config, &air, &proof, &[], Some(&pp_vk))
}
