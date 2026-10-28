use p3_air::{Air, AirBuilder, BaseAir, NextRowTrace, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{ProvingError, StarkConfig, VerificationError, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

/// A counter: the first row is zero and each row adds one to the previous.
///
/// The transition reads column 1 of the next row.
/// Column 0 is a copy of column 1, so the declaration can omit exactly one read column.
struct CounterAir {
    /// The main columns the AIR declares it reads on the next row.
    declared_next: Vec<usize>,
}

impl<F> BaseAir<F> for CounterAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        self.declared_next.clone()
    }
}

impl<AB: AirBuilder> Air<AB> for CounterAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let copy = main.current(0).unwrap();
        let local = main.current(1).unwrap();
        let next = main.next(1).unwrap();
        builder.assert_eq(copy, local);
        builder.when_first_row().assert_zero(local);
        builder
            .when_transition()
            .assert_eq(next, local + AB::Expr::ONE);
    }
}

fn make_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = Pcs::new(Dft::default(), val_mmcs, fri_params);
    MyConfig::new(pcs, Challenger::new(perm))
}

/// Eight rows counting from zero, each value written to both columns.
fn counter_trace() -> RowMajorMatrix<Val> {
    let values = (0..8u64).flat_map(|i| [Val::from_u64(i); 2]).collect();
    RowMajorMatrix::new(values, 2)
}

#[test]
fn prover_rejects_an_undeclared_next_row_read() {
    let config = make_config();

    // Fixture state: the trace satisfies every constraint on its real rows.
    //
    //     row:      0  1  2  ...  7
    //     col 0/1:  0  1  2  ...  7
    //
    // Mutation: the AIR declares no next-row column, yet its transition reads column 1.
    let air = CounterAir {
        declared_next: vec![],
    };

    // Invariant: the prover reports the read before committing to anything.
    let result = prove(&config, &air, counter_trace(), &[]);
    assert!(matches!(
        result,
        Err(ProvingError::UndeclaredNextRowRead {
            air: None,
            trace: NextRowTrace::Main,
            column: 1,
        })
    ));
}

#[test]
fn verifier_rejects_an_undeclared_next_row_read() {
    let config = make_config();

    // Fixture state: an honest proof of the counter, with its next-row read declared.
    let honest = CounterAir {
        declared_next: vec![1],
    };
    let proof = prove(&config, &honest, counter_trace(), &[]).unwrap();
    assert!(matches!(verify(&config, &honest, &proof, &[]), Ok(())));

    // Mutation: the verifier's AIR declares only column 0, which no transition reads.
    //
    //     declared next row:  {0}
    //     read next row:      {1}   <- undeclared
    let misdeclared = CounterAir {
        declared_next: vec![0],
    };

    // Invariant: the verifier names the read before any opening check.
    let result = verify(&config, &misdeclared, &proof, &[]);
    assert!(matches!(
        result,
        Err(VerificationError::UndeclaredNextRowRead {
            air: None,
            trace: NextRowTrace::Main,
            column: 1,
        })
    ));
}
