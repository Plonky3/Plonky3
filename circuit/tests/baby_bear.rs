//! The same verifier circuit over BabyBear, whose Poseidon2 uses a degree-7 S-box.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{
    BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL, BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    BABYBEAR_POSEIDON2_RC_16_INTERNAL, BabyBear, GenericPoseidon2LinearLayersBabyBear,
    Poseidon2BabyBear, default_babybear_poseidon2_16,
};
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch, verify_batch_in};
use p3_challenger::DuplexChallenger;
use p3_circuit::{CircuitArith, Poseidon2Chip};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2_air::RoundConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
type Pcs = TwoAdicFriPcs<F, Radix2DitParallel<F>, ValMmcs, ChallengeMmcs>;
type Config = StarkConfig<Pcs, EF, DuplexChallenger<F, Perm, 16, 8>>;
type Chip = Poseidon2Chip<F, GenericPoseidon2LinearLayersBabyBear, Perm, 16, 7, 1, 4, 13>;

/// `x -> x^2 + 1` per row, with the first and last values public.
#[derive(Clone, Copy, Debug)]
struct Squares;

impl<T> BaseAir<T> for Squares {
    fn width(&self) -> usize {
        1
    }

    fn num_public_values(&self) -> usize {
        2
    }
}

impl<AB: AirBuilder> Air<AB> for Squares {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.current_slice()[0], main.next_slice()[0]);
        let publics = builder.public_values().to_vec();
        builder.when_first_row().assert_eq(local, publics[0]);
        builder
            .when_transition()
            .assert_eq(next, local * local + AB::Expr::ONE);
        builder.when_last_row().assert_eq(local, publics[1]);
    }
}

#[test]
fn a_baby_bear_proof_verifies_inside_a_baby_bear_circuit() {
    let perm = default_babybear_poseidon2_16();
    let mmcs = ValMmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let fri = FriParameters {
        num_queries: 2,
        ..FriParameters::new_testing(ChallengeMmcs::new(mmcs.clone()), 0)
    };
    let config = Config::new(
        Pcs::new(Radix2DitParallel::default(), mmcs, fri),
        DuplexChallenger::new(perm.clone()),
    );

    // The inner proof: eight squarings from 2.
    let mut values = vec![F::TWO];
    for _ in 1..8 {
        let last = values[values.len() - 1];
        values.push(last * last + F::ONE);
    }
    let publics = [vec![values[0], values[7]]];
    let trace = RowMajorMatrix::new(values, 1);
    let instances = vec![StarkInstance {
        air: &Squares,
        trace: &trace,
        public_values: publics[0].clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances).unwrap();
    let proof = prove_batch(&config, &instances, &prover_data).unwrap();

    // The verifier circuit, built over BabyBear with its own Poseidon2 table.
    let chip = Chip::new(
        perm,
        RoundConstants::new(
            BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            BABYBEAR_POSEIDON2_RC_16_INTERNAL,
            BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        ),
    );
    let mut arith = CircuitArith::<F, EF, _>::new(chip);
    let lifted = arith.public_inputs(&publics);
    verify_batch_in(
        &mut arith,
        &config,
        &[Squares],
        &proof,
        &lifted,
        &prover_data.common,
    )
    .expect("the inner proof verifies inside the circuit");

    // The circuit's own proof verifies natively.
    let circuit = arith.build();
    let instances = circuit.instances::<Config>();
    let circuit_data = ProverData::from_instances(&config, &instances).unwrap();
    let outer = prove_batch(&config, &instances, &circuit_data).unwrap();
    verify_batch(
        &config,
        &circuit.airs,
        &outer,
        &circuit.public_values(),
        &circuit_data.common,
    )
    .expect("the circuit proof verifies");
}
